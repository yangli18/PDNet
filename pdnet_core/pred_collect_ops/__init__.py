from .pred_collect import CollectConcat, CollectMerge


def build_pred_collect_module(concat=True, **kwargs):
    if concat:
        return CollectConcat(**kwargs)
    return CollectMerge(**kwargs)