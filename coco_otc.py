from dataclasses import dataclass
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.builder import DATASETS
import numpy as np
# from src.extensions.metrics.ot_cost import get_ot_cost, get_cmap
from copy import deepcopy
import pdb
import json
import time
import os.path as osp
from mmcv.utils import get_logger
import matplotlib.pyplot as plt
import seaborn as sns
import neptune.new as neptune
from neptune.new.types import File
from mmcv.runner.dist_utils import master_only
from typing import Callable, Sequence, Tuple, Callable, Union
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
import ot
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist

N_COCOCLASSES = 80


def bbox_gious(
    bboxes1: npt.ArrayLike,
    bboxes2: npt.ArrayLike,
    eps: float = 1e-6,
    use_legacy_coordinate: bool = False,
) -> npt.ArrayLike:
    """Calculate the generalized ious between each bbox of bboxes1 and bboxes2.
    Args:
        bboxes1 (ndarray): Shape (n, 4) # [[x1, y1, x2, y2], ...]
        bboxes2 (ndarray): Shape (k, 4) # [[x1, y1, x2, y2], ...]
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Note when function is used in `VOCDataset`, it should be
            True to align with the official implementation
            `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar`
            Default: False.
    Returns:
        gious (ndarray): Shape (n, k)
    """

    if not use_legacy_coordinate:
        extra_length = 0.0
    else:
        extra_length = 1.0

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    gious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return gious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        gious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + extra_length) * (
        bboxes1[:, 3] - bboxes1[:, 1] + extra_length
    )
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + extra_length) * (
        bboxes2[:, 3] - bboxes2[:, 1] + extra_length
    )
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + extra_length, 0) * np.maximum(
            y_end - y_start + extra_length, 0
        )

        union = area1[i] + area2 - overlap
        union = np.maximum(union, eps)
        ious = overlap / union

        # Finding the coordinate of smallest enclosing box
        x_min = np.minimum(bboxes1[i, 0], bboxes2[:, 0])
        y_min = np.minimum(bboxes1[i, 1], bboxes2[:, 1])
        x_max = np.maximum(bboxes1[i, 2], bboxes2[:, 2])
        y_max = np.maximum(bboxes1[i, 3], bboxes2[:, 3])
        hull = (x_max - x_min + extra_length) * (y_max - y_min + extra_length)

        gious[i, :] = ious - (hull - union) / hull

    if exchange:
        gious = gious.T

    return gious


def add_label(result: Sequence[Sequence]) -> npt.ArrayLike:
    labels = [[i] * len(r) for i, r in enumerate(result)]
    labels = np.hstack(labels)
    return np.hstack([np.vstack(result), labels[:, None]])


def cost_func(x, y, mode: str = "giou", alpha: float = 0.8):
    """Calculate a unit cost

    Args:
        x (np.ndarray): a detection [x1, y1, x2, y2, s, l]. s is a confidence value, and l is a classification label.
        y (np.ndarray): a detection [x1, y1, x2, y2, s, l]. s is a confidence value, and l is a classification label.
        mode (str, optional): Type of IoUs. Defaults to "giou" (Generalized IoU).
        alpha (float, optional): weights to balance localization and classification errors. Defaults to 0.8.

    Returns:
        float: a unit cost
    """
    giou_val = bbox_gious(x[:4][None, :], y[:4][None, :])  # range [-1, 1]
    loc_cost = 1 - (giou_val + 1) * 0.5  # normalized to [0, 1]
    l_x, l_y = x[-1], y[-1]
    if l_x == l_y:
        cls_cost = np.abs(x[-2] - y[-2])
    else:
        cls_cost = x[-2] + y[-2]
    cls_cost *= 0.5  # normalized to [0, 1]

    return alpha * loc_cost + (1 - alpha) * cls_cost


def get_cmap(
    a_result: Sequence[npt.ArrayLike],
    b_result: Sequence[npt.ArrayLike],
    alpha: float = 0.8,
    beta: float = 0.4,
    mode="giou",
) -> Tuple[npt.ArrayLike]:
    """Calculate cost matrix

    Args:
        a_result ([type]): detections
        b_result ([type]): detections
        mode (str, optional): [description]. Defaults to "giou".

    Returns:
        dist_a (np.array): (N+1,) array. distribution over detections.
        dist_b (np.array): (M+1,) array. distribution over detections.
        cost_map:
    """
    a_result = add_label(a_result)
    b_result = add_label(b_result)
    n = len(a_result)
    m = len(b_result)

    cost_map = np.zeros((n + 1, m + 1))

    metric = lambda x, y: cost_func(x, y, alpha=alpha, mode=mode)
    cost_map[:n, :m] = cdist(a_result, b_result, metric)

    dist_a = np.ones(n + 1)
    dist_b = np.ones(m + 1)

    # cost for dummy demander / supplier
    cost_map[-1, :] = beta
    cost_map[:, -1] = beta
    dist_a[-1] = m
    dist_b[-1] = n

    return dist_a, dist_b, cost_map


def postprocess(M: npt.ArrayLike, P: npt.ArrayLike) -> float:
    """drop dummy to dummy costs, normalize the transportation plan, and return total cost

    Args:
        M (npt.ArrayLike): correction cost matrix
        P (npt.ArrayLike)): optimal transportation plan matrix

    Returns:
        float: _description_
    """
    P[-1, -1] = 0
    P /= P.sum()
    total_cost = (M * P).sum()
    return total_cost


def get_ot_cost(
    a_detection: list,
    b_detection: list,
    costmap_func: Callable,
    return_matrix: bool = False,
) -> Union[float, Tuple[float, dict]]:
    """[summary]

    Args:
        a_detection (list): list of detection results. a_detection[i] contains bounding boxes for i-th class.
        Each element is numpy array whose shape is N x 5. [[x1, y1, x2, y2, s], ...]
        b_detection (list): ditto
        costmap_func (callable): a function that takes a_detection and b_detection as input and returns a unit cost matrix
    Returns:
        [float]: optimal transportation cost
    """

    if sum(map(len, a_detection)) == 0:
        if sum(map(len, b_detection)) == 0:
            return 0

    a, b, M = costmap_func(a_detection, b_detection)
    P = ot.emd(a, b, M)
    total_cost = postprocess(M, P)

    if return_matrix:
        log = {"M": M, "a": a, "b": b}
        return total_cost, log
    else:
        return total_cost


def count_items(items):
    ns = []
    for x in items:
        if x is None:
            n = 0
        else:
            n = sum(map(len, x))
        ns.append(n)
    return ns


def get_stats(ot_costs, gts, results):
    mean = np.mean(ot_costs)
    std = np.std(ot_costs)

    n_gts = count_items(gts)
    n_preds = count_items(results)

    cov_gts = np.cov(ot_costs, n_gts)[0, 1]
    cov_preds = np.cov(ot_costs, n_preds)[0, 1]

    return {
        "mean": mean,
        "std": std,
        "cov_n-gts": cov_gts,
        "cov_n-preds": cov_preds,
    }


def draw_stats(ot_costs, gts, results):
    n_gts = count_items(gts)
    n_preds = count_items(results)
    figures = {}

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    sns.kdeplot(x=n_gts, y=ot_costs, fill=True, cmap="rocket", ax=axes[0])
    # axes[0].scatter(n_gts, ot_costs)
    axes[0].set_title("otc vs # GTs")
    sns.kdeplot(x=n_preds, y=ot_costs, fill=True, cmap="rocket", ax=axes[1])
    # axes[1].scatter(n_preds, ot_costs)
    axes[1].set_title("otc vs # Preds")
    figures["otc_vs_num_bb"] = fig

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    axes[0].hist(n_gts, bins=10)
    axes[0].set_title("# Ground truth boudning boxes")
    axes[1].hist(n_preds, bins=10)
    axes[1].set_title("# Prediction boudning boxes")
    figures["dist_n_bb"] = fig

    fig = plt.figure()
    plt.hist(ot_costs, bins=10)
    plt.title("OTC Distribution")
    figures["dist_otc"] = fig

    fig_src = {"ot_costs": ot_costs, "n_gts": n_gts, "n_preds": n_preds}

    return figures, fig_src


def write2json(ot_costs, file_names):
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    json_file = osp.join(f"tmp/otc_{timestamp}.json")
    data = [(f_name, c) for f_name, c in zip(file_names, ot_costs)]
    json.dump(data, open(json_file, "w"))


def eval_ot_costs(gts, results, cmap_func):
    return [get_ot_cost(x, y, cmap_func) for x, y in zip(gts, results)]


@DATASETS.register_module()
class CocoOtcDataset(CocoDataset):
    def __init__(
        self,
        ann_file,
        pipeline,
        classes=None,
        data_root=None,
        img_prefix="",
        seg_prefix=None,
        proposal_file=None,
        test_mode=False,
        filter_empty_gt=True,
        nptn_project_id="",
        nptn_run_id="",
        nptn_metadata_suffix="",
    ):

        super().__init__(
            ann_file,
            pipeline,
            classes=classes,
            data_root=data_root,
            img_prefix=img_prefix,
            seg_prefix=seg_prefix,
            proposal_file=proposal_file,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt,
        )

        self.nptn_project_id = nptn_project_id
        self.nptn_run_id = nptn_run_id
        self.nptn_on = False

        if (nptn_project_id != "") and (nptn_run_id != ""):
            self.nptn_metadata_suffix = nptn_metadata_suffix
            self.nptn_on = True

    def evaluate(
        self,
        results,
        metric="bbox",
        logger=None,
        jsonfile_prefix=None,
        classwise=False,
        proposal_nums=(100, 300, 1000),
        iou_thrs=None,
        metric_items=None,
        eval_map=True,
        otc_params=[("alpha", 0.5), ("beta", 0.6)],
    ):
        """Evaluate predicted bboxes. Overide this method for your measure.

        Args:
            results ([type]): outputs of a detector
            metric (str, optional): [description]. Defaults to "bbox".
            logger ([type], optional): [description]. Defaults to None.
            jsonfile_prefix ([type], optional): [description]. Defaults to None.
            classwise (bool, optional): [description]. Defaults to False.
            proposal_nums (tuple, optional): [description]. Defaults to (100, 300, 1000).
            iou_thrs ([type], optional): [description]. Defaults to None.
            metric_items ([type], optional): [description]. Defaults to None.
            eval_map (bool): Whether to evaluating mAP
            otc_params (list): OC-cost parameters.
                                alpha (lambda in the paper): balancing localization and classification costs.
                                beta: cost of extra / missing detections.
                                Defaults to [("alpha", 0.5), ("beta", 0.6)]

        Returns:
            dict[str, float]: {metric_name: metric_value}
        """
        if eval_map:
            eval_results = super().evaluate(
                results,
                metric=metric,
                logger=logger,
                jsonfile_prefix=jsonfile_prefix,
                classwise=classwise,
                proposal_nums=proposal_nums,
                iou_thrs=iou_thrs,
                metric_items=metric_items,
            )
        else:
            eval_results = {}

        otc_params = {k: v for k, v in otc_params}
        mean_otc = self.eval_OTC(results, **otc_params)
        eval_results["mOTC"] = mean_otc

        if self.nptn_on:
            self.upload_eval_results(eval_results)

        return eval_results

    def get_gts(self):
        gts = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            gt = self._ann2detformat(ann_info)
            if gt is None:
                gt = [
                    np.asarray([]).reshape(0, 5)
                    for _ in range(len(self.CLASSES))
                ]
            gts.append(gt)
        return gts

    @master_only
    def upload_eval_results(self, eval_results):
        nptn_run = neptune.init(
            project=self.nptn_project_id,
            run=self.nptn_run_id,
            mode="sync",
            capture_hardware_metrics=False,
        )
        for k, v in eval_results.items():
            nptn_run[f"evaluation/summary/{k}/{self.nptn_metadata_suffix}"] = v
        nptn_run.stop()

    @master_only
    def upload_otc_results(self, ot_costs, gts, results):
        nptn_run = neptune.init(
            project=self.nptn_project_id,
            run=self.nptn_run_id,
            mode="sync",
            capture_hardware_metrics=False,
        )

        file_names = [x["file_name"] for x in self.data_infos]
        otc_per_img = json.dumps(list(zip(file_names, ot_costs)))
        nptn_run[f"evaluation/otc/per_img/{self.nptn_metadata_suffix}"].upload(
            File.from_content(otc_per_img, extension="json")
        )

        for k, v in get_stats(ot_costs, gts, results).items():
            nptn_run[
                f"evaluation/otc/stats/{k}/{self.nptn_metadata_suffix}"
            ] = v

        figs, fig_src = draw_stats(ot_costs, gts, results)
        for fig_name, fig in figs.items():
            nptn_run[
                f"evaluation/figs/{fig_name}/{self.nptn_metadata_suffix}"
            ].upload(File.as_image(fig))
            fig.savefig(f"tmp/{fig_name}.pdf", bbox_inches="tight")
            nptn_run[
                f"evaluation/figs/pdfs/{fig_name}/{self.nptn_metadata_suffix}"
            ].upload(f"tmp/{fig_name}.pdf")

        nptn_run.stop()

    def eval_OTC(
        self,
        results,
        alpha=0.8,
        beta=0.4,
        get_average=True,
    ):
        gts = self.get_gts()
        cmap_func = lambda x, y: get_cmap(
            x, y, alpha=alpha, beta=beta, mode="giou"
        )
        tic = time.time()
        ot_costs = eval_ot_costs(gts, results, cmap_func)
        toc = time.time()
        print("OTC DONE (t={:0.2f}s).".format(toc - tic))

        if self.nptn_on:
            self.upload_otc_results(ot_costs, gts, results)

        if get_average:
            mean_ot_costs = np.mean(ot_costs)
            return mean_ot_costs
        else:
            return ot_costs

    def evaluate_gt(
        self,
        bbox_noise_level=None,
        **kwargs,
    ):

        gts = self.get_gts()
        n = len(gts)
        for i in range(n):
            gt = gts[i]
            if gt is None:
                gts[i] = [np.asarray([]).reshape(0, 5) for _ in self.CLASSES]
                continue
            for bbox in gt:
                if len(bbox) == 0:
                    continue

                w = bbox[:, 2] - bbox[:, 0]
                h = bbox[:, 3] - bbox[:, 1]
                shift_x = (
                    w * bbox_noise_level * np.random.choice((-1, 1), w.shape)
                )
                shift_y = (
                    h * bbox_noise_level * np.random.choice((-1, 1), h.shape)
                )
                bbox[:, 0] += shift_x
                bbox[:, 2] += shift_x
                bbox[:, 1] += shift_y
                bbox[:, 3] += shift_y
        return self.evaluate(gts, **kwargs)

    def _ann2detformat(self, ann_info):
        """convert annotation info of CocoDataset into detection output format.

        Parameters
        ----------
        ann : list[dict]
            ground truth annotation. each item in the list correnponds to an instance.
            >>> ann_info[i].keys()
            dict_keys(['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'])
        Returns
        -------
        bboxes : list[numpy]
            list of bounding boxes with confidence score.
            bboxes[i] contains bounding boxes of instances of class i.
        """
        if len(ann_info) == 0:
            return None

        bboxes = [[] for _ in range(len(self.cat2label))]

        for ann in ann_info:
            if ann.get("ignore", False) or ann["iscrowd"]:
                continue
            c_id = ann["category_id"]
            x1, y1, w, h = ann["bbox"]

            bboxes[self.cat2label[c_id]].append([x1, y1, x1 + w, y1 + h, 1.0])

        np_bboxes = []
        for x in bboxes:
            if len(x):
                np_bboxes.append(np.asarray(x, dtype=np.float32))
            else:
                np_bboxes.append(
                    np.asarray([], dtype=np.float32).reshape(0, 5)
                )
        return np_bboxes
