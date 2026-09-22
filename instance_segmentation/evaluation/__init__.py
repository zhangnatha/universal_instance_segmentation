"""框架无关的掩码匹配、指标计算与评估报告模块。"""

from importlib import import_module

_MODULES = {
    "mask_iou": ("metrics", "mask_iou"),
    "match_instances": ("metrics", "match_instances"),
    "precision_recall_f1": ("metrics", "precision_recall_f1"),
    "evaluate_predictions": ("metrics", "evaluate_predictions"),
    "match_class_instances": ("compare_json", "match_class_instances"),
    "finalize_stats": ("compare_json", "finalize_stats"),
    "init_stats": ("compare_json", "init_stats"),
}


def __getattr__(name):
    try:
        module_name, attr_name = _MODULES[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(f"{__name__}.{module_name}"), attr_name)
    globals()[name] = value
    return value


__all__ = list(_MODULES)
