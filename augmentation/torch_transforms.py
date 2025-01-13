import ast
import torchvision.transforms as transforms
from ionpy.experiment.util import absolute_import


def initialize_transforms(transform_list):
    """Initializes the transforms pipeline."""
    if transform_list is None:
        return None
    
    # Initialize the transforms pipeline
    transforms_pipeline = []
    for transform in transform_list:
        if isinstance(transform, dict):
            transform_name = list(transform.keys())[0]
            transform_args = transform[transform_name]
            # Transform args is a dictionary, where some of the values could potentially
            # be tuples that are currently strings (artificat from yaml input) so we need to
            # parse them as tuples
            for key, value in transform_args.items():
                if isinstance(value, str):
                    if value.startswith("(") and value.endswith(")"):
                        transform_args[key] = ast.literal_eval(value)
            transform_instance = absolute_import(transform_name)(**transform_args)
        else:
            transform_instance = absolute_import(transform)()
        transforms_pipeline.append(transform_instance)

    return transforms.Compose(transforms_pipeline)