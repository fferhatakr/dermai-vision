import torch
import yaml
from src.datalar.dataset import TripletDermaDataset

def test_triplet_dataset_output():
    """
    This test robot checks whether our data set actually provides 3 images
    and whether the image dimensions are correct.
    """

    with open("configs/train_config.yaml","r",encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_path = config['data']['data_path']
    dataset = TripletDermaDataset(data_path)

    anchor , positive , negative = dataset[0]


    #The word assert means "guarantee this, or throw an error".
    assert isinstance(anchor,torch.Tensor), "Anchor is not a Tensor!"
    assert isinstance(positive,torch.Tensor),"It is not a 'Positive' Tensor!"
    assert isinstance(negative,torch.Tensor), "It's not a 'negative' Tensor!"

    expected_size = torch.Size([3,224,224])
    assert anchor.shape == expected_size, f"Anchor size is incorrect! Output: {anchor.shape}"
    assert positive.shape == expected_size, f"The positive dimension is incorrect! Output: {positive.shape}"
    assert negative.shape == expected_size, f"Negative dimension is incorrect! Output: {negative.shape}"




    