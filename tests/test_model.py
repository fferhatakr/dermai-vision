"""
Model Inference Tests

What these tests do:
- Check that the model produces output in the correct shape
- Verify that the probabilities after softmax are between 0 and 1 and sum to 1
- Test that the backbone selection works

How to run:
pytest test/test_model.py -v

Note: These tests do not require a GPU; they run on the CPU.
"""

import pytest
import torch
import sys
import os

sys.path.append(os.getcwd())
from configs.config import cfg



class TestModelArchitecture:
    """
    Tests that the model architecture is functioning correctly.

    "Correct functioning" = accepting input of the correct size and producing output of the correct size.
    This is the fundamental contract of a CNN:

        Input: (batch_size, 3, 300, 300) → 3-channel, 300x300-pixel image
        Output: (batch_size, 8) → raw scores (logits) for 8 classes
    """


    def test_efficientnet_output_shape(self):
        from src.architectures.vision_model import DermaScanModelV3
        model = DermaScanModelV3(num_classes=cfg.NUM_CLASSES, pretrained=False)

        model.eval()

        dummy_input = torch.rand(1 ,3 , cfg.IMAGE_SIZE,cfg.IMAGE_SIZE)

        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (1, cfg.NUM_CLASSES), \
            f"Expected (1 , {cfg.NUM_CLASSES}) got {output.shape}"
        
    
    def test_convnext_output_shape(self):
        """ConvNeXt-Tiny must also support the same contract.
        Different architecture but the same input/output dimensions.

        If the ConvNeXt classifier head is implemented incorrectly,
        the output will be something like (1, 768) → the loss cannot be calculated."""

        from src.architectures.vision_model import DermaScanModelV4

        model = DermaScanModelV4(num_classes=cfg.NUM_CLASSES , pretrained=False)
        model.eval()
        dummy_input = torch.rand(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (1, cfg.NUM_CLASSES), \
            f"Expected ( 1, {cfg.NUM_CLASSES}), got {output.shape}"
        
    def test_batch_inference(self):
        """
        Does batch inference work? (batch_size > 1)

        We use batch_size=16 during training. If the model only
        accepts a single image (as is the case with some incorrectly written models),
        it crashes when training begins.

        This test checks using a batch of four images.
        """
        from src.architectures.vision_model import DermaScanModelV3

        model = DermaScanModelV3(num_classes=cfg.NUM_CLASSES, pretrained=False)
        model.eval()

        batch_size=4

        dummy_input = torch.randn(batch_size, 3, cfg.IMAGE_SIZE,cfg.IMAGE_SIZE)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (batch_size, cfg.NUM_CLASSES)

    def test_softmax_prob(self):
        """
        Are the probabilities after softmax between 0 and 1, and does their sum equal 1?

        WHY IT MATTERS:
        In the API, we take prob_mel = softmax(logits)[0].
        Threshold: if prob_mel > 0.11, then "RISKY"
        If the softmax is corrupted, it produces nonsensical values such as -0.3 or 5.7
        → the threshold logic works completely incorrectly.
        """
        
        from src.architectures.vision_model import DermaScanModelV3
        import torch.nn.functional as F
        
        model = DermaScanModelV3(num_classes= cfg.NUM_CLASSES,pretrained=False)
        model.eval()

        dummy_input = torch.randn(1, 3, cfg.IMAGE_SIZE,cfg.IMAGE_SIZE)

        with torch.no_grad():

            logits = model(dummy_input)

            probs = F.softmax(logits, dim=1)

        assert (probs >=0).all(), "Negative prob found"
        assert (probs <=1).all(), "Probability > 1 found"

        prob_sum = probs.sum(dim=1)
        assert torch.allclose(prob_sum, torch.ones(1),atol=1e-5), \
            f"Probabilities sum to {prob_sum.item()}, expected 1.0"
        

class TestLightningModule:
    """
    Tests that the PyTorch Lightning module is working correctly.

    DermatologLightning is our main training class.
    It contains the model, metrics, loss function and optimiser.
    If this class is broken, nothing will work.
    """
    def test_trainer_core_init(self):    
        """
        Is DermatologLightning being created successfully?

        Sometimes an import error, a parameter error or a bug within the __init__
        method prevents the class from being created. This test performs the most basic check.

        assert model is not None → Has the model object been created?
        """
        from engine.trainer_core import DermatologLightning
        model= DermatologLightning(backbone="efficientnet_b3")
        assert model is not None

    def test_trainer_core_with_class_w(self):
        """
        Are the class weights being recorded correctly when we assign them?

        Class weights = class weights. We assign higher weights to classes with fewer examples
        so that the model learns them as well.
        Example: There are 239 examples in the DF class and 12,875 in the NV class → DF’s weight
        should be much higher.
        
        """

        from engine.trainer_core import DermatologLightning

        weights = torch.ones(cfg.NUM_CLASSES)
        model = DermatologLightning(
            class_weights=weights,
            backbone="efficientnet_b3"
        )

        assert model.class_weights is not None

        assert model.class_weights.shape[0] == (cfg.NUM_CLASSES)

    def test_separete_train_val_metrics(self):
        """Are the train and val metrics SEPARATE OBJECTS?

        BACKGROUND:
        Previously, a single `self.accuracy` object was used for both train and val.
        TorchMetrics keeps an internal counter:

        In train: 7000 correct / 10000 total accumulated
        Val started: 500 correct / 2700 total added
        Result: (7000+500) / (10000+2700) = 59% → BUT actual val = 500/2700 = 18%

        This is why val_acc was showing 100% — train data was getting mixed in!

        FIX: self.train_accuracy and self.val_accuracy are separate objects.
        This test ensures the FIX hasn’t been corrupted.

        The "is not" operator: do the two variables point to the SAME OBJECT?
        a = [1,2,3]
        b = [1,2,3]
        a == b → True (values are the same)
        a is b → False (different objects)
        a is not b → True (different objects — this is what we want!)"""


        from engine.trainer_core import DermatologLightning

        model = DermatologLightning(backbone="efficientnet_b3")
        assert model.train_accuracy is not model.val_accuracy, \
            "train_acc and val_acc must be different obhects"
        
    def test_forward(self):
        """Does the forward pass work via the Lightning module?

        DermatologLightning.forward() calls self.model(x).
        If forward() is not defined correctly, model(input) will raise an error.

        This test verifies that both DermatologLightning and the DermaScanModelV3 within it
        work together."""


        from engine.trainer_core import DermatologLightning

        model = DermatologLightning(backbone="efficientnet_b3")
        model.eval()

        dummy_imput = torch.randn(1, 3, cfg.IMAGE_SIZE ,cfg.IMAGE_SIZE)

        with torch.no_grad():
            output = model(dummy_imput)

        assert output.shape == (1, cfg.NUM_CLASSES)
