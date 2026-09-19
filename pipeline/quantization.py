
QUANTIZZERS = {
    "RK-Means": "pipeline.quantization.rkmeans.RKMeansQuantizer"}

def recommender_factory(model_name: str, cfg: dict):
    """Factory function to create a recommender model based on the model name."""
    if model_name not in QUANTIZZERS:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(QUANTIZZERS.keys())}")
    
    module_path, class_name = QUANTIZZERS[model_name].rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)
    
    return model_class(cfg)

