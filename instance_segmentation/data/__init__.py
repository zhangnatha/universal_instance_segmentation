"""数据集格式、划分、预处理、数据增强与重采样模块。

大型图像依赖库仅在调用相应功能时按需导入，
以确保 ``instance-seg capabilities`` 及演练（dry-run）模式无需额外依赖即可运行。
"""

from importlib import import_module

_MODULES = {
    "Annotation": ("labelme_core", "Annotation"),
    "discover_samples": ("labelme_core", "discover_samples"),
    "parse_labelme": ("labelme_core", "parse_labelme"),
    "SplitManifest": ("split", "SplitManifest"),
    "create_split_manifest": ("split", "create_split_manifest"),
    "split_samples": ("split", "split_samples"),
    "AnnotationItem": ("dataset", "AnnotationItem"),
    "DatasetManager": ("dataset", "DatasetManager"),
    "ImageSample": ("dataset", "ImageSample"),
    "AugmentationPipeline": ("augmentation", "AugmentationPipeline"),
    "PreprocessingPipeline": ("preprocessing", "PreprocessingPipeline"),
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
