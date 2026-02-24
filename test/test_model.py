import torch
import yaml
from src.lightning_model import TripletLightning


def test_triplet_model_output_shape():
    """
    This test checks whether our model takes images and converts t
    hem into a mathematical vector (embedding) of the correct size.
    """


    test_lr = 0.00001
    test_margin = 1.0

    model = TripletLightning(learning_rate=test_lr,margin_value=test_margin)

    model.eval()

    fake_image = torch.randn(1,3,224,224)

    result = model(fake_image)

    expected_size = torch.Size([1,960])
    
    assert result.shape == expected_size, f"The model produced an output of the wrong size! Output: {result.shape}"