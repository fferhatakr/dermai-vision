import yaml
import os

file_path = "configs/inference_config.yaml"
def test_inference_config_loads():
    with open(file_path,"r",encoding="utf-8") as file:
        config = yaml.safe_load(file)
    """
    This test checks that our inference_config.yaml file can be read correctly
    and that the settings within it (and the model file) are valid.
    """
    


    assert config['search']['top_k'] == 5 ,"The value of top_k is not 5!"
    assert config['search']['similarity_threshold'] == 0.89, "Threshold value is incorrect!"

    model_path = config['model']['path']
    assert os.path.exists(str(file_path)), f"Model file not found: {model_path}"