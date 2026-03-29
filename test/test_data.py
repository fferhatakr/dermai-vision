import pytest
import torch
import numpy as np
from PIL import Image
import sys
import os

sys.path.append(os.getcwd())
from configs.config import cfg



class TestTransforms:
    """
    Tests whether data transforms are working correctly.
    
    Transform = a sequence of operations that converts the image into the format expected by the model.
    
    Train transform: Resize → RandomFlip → Rotate → ColorJitter → ... → ToTensor → Normalize
    Val transform:   Resize → ToTensor → Normalize (no augmentation)
    
    ToTensor: PIL Image (H,W,3) uint8 [0-255] → Tensor (3,H,W) float [0-1]
    Normalize: (pixel - mean) / std → approximately [-2.5, +2.5]
    """

    def _make_dummy_image(self, size=(600,450)):

        """
        Creates a dummy PIL image for testing.
        
        np.random.randint(0, 255, ...) → random pixel values between 0 and 255
        dtype=np.uint8 → pixel values are 8-bit integers (standard image format)
        (*size, 3) → (600, 450, 3) = 600 height, 450 width, 3 channels (RGB)
        Image.fromarray() → converts the NumPy array into a PIL image
        """
        return Image.fromarray(
            np.random.randint(0,255,(*size,3),dtype=np.uint8)
        )
    
    def test_train_Transform_output_shape(self):

        """
        Does the train transform produce the correct tensor shape?
        
        Input: PIL image of any size (e.g., 600x450)
        Output: (3, 300, 300) tensor
        
        3 = RGB channels
        300, 300 = Fixed dimensions after resizing
        
        If the transform forgets to resize, the output becomes (3, 600, 450)
        → The model throws an error because it expects (3, 300, 300).
        """
        from src.dataloader.image_dataset import train_transforms

        img=self._make_dummy_image()
        tensor= train_transforms(img)

        assert isinstance(tensor,torch.Tensor)

        assert tensor.shape == (3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), \
            f"Expected (3, {cfg.IMAGE_SIZE},{cfg.IMAGE_SIZE}), got {tensor.shape}"


    def test_val_transform_output_shape(self):
        from src.dataloader.image_dataset import val_transforms

        img = self._make_dummy_image()
        tensor = val_transforms(img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)

    def test_val_transform_is_deterministic(self):
        """
        Is the validation transformation deterministic? (Same input → same output)
        
        WHY IT MATTERS:
        There should be no augmentation in validation. If RandomFlip is used in validation,
        the same image will receive a different prediction each time → metrics are unreliable.
        
        The train transform includes RandomFlip, Rotation, etc. → NOT deterministic.
        The val transform includes only Resize, ToTensor, Normalize → MUST be deterministic.
        
        torch.equal(a, b) → Are the two tensors EXACTLY the same? (every element is equal)
        """
        from src.dataloader.image_dataset import val_transforms
        img = self._make_dummy_image()

        tensor1 = val_transforms(img)
        tensor2 = val_transforms(img)


        assert torch.equal(tensor1, tensor2), \
            "val transform should be deterministic"
        
    def test_train_transform_is_normalized(self):

        """
        Is the train transform normalized?
        
        BEFORE normalization: pixel values range from 0.0 to 1.0 (after ToTensor)
        AFTER normalization: pixel values range from approximately -2.5 to +2.5
        
        How it works: (pixel - mean) / std
        Example: (0.5 - 0.485) / 0.229 = 0.065 → positive
                (0.0 - 0.485) / 0.229 = -2.11 → negative
        
        If not normalized, all values are between 0 and 1 → NO negative values.
        This test verifies normalization by checking for the presence of negative values.
        """
        from src.dataloader.image_dataset import train_transforms
        img = self._make_dummy_image()
        tensor = train_transforms(img)

        assert tensor.min() < 0, "Tensor should have negative values after normalization"
        assert tensor.max() > 0, "Tensor should have positive values after normalization"


class TestMetadataCSV:
    """
    Tests the structure and content of the metadata CSV.
    
    This CSV is the backbone of the training pipeline:
    - image column → matches files in ImageFolder
    - targets column → class label (0–7)
    - lesion_id column → Grouping for StratifiedGroupKFold
    
    If any of these are corrupted → incorrect split, incorrect label, data leakage.
    """

    @pytest.fixture
    def metadata(self):
        """
        Upload the CSV. If the file is missing, skip the test (there may be no data in CI).
        
        @pytest.fixture → This function is not a test; it provides data to tests.
        When the following test functions take a “metadata” parameter,
        pytest automatically calls this fixture.
        """

        import pandas as pd 
        if not os.path.exists(cfg.CSV_PATH):
            pytest.skip(f"CSV not found: {cfg.CSV_PATH}")
        return pd.read_csv(cfg.CSV_PATH)
    
    def test_required_columns_exist(self, metadata):
        """
        Are the required columns present?
        
        The 'metadata' parameter → comes from the fixture above.
        pytest automatically binds this; you don't need to call it.
        
        'metadata.columns' → A list of the CSV's column names
        """

        required = ['image', 'targets', 'lesion_id']
        for col in required:
            assert col in metadata.columns, f"Missing column: {col}"

    def test_targets_are_Valid(self, metadata):
        """
                Are the target values between 0 and 7?
        
        If there are invalid values in 'targets', such as 8 or -1:
        - They do not match the labels from `ImageFolder`
        - The 'class_weights' calculation is corrupted
        - The model learns the wrong class
        """
        assert metadata['targets'].min() >= 0
        assert metadata['targets'].max() <= cfg.NUM_CLASSES -1


    def test_all_classes_represented(self , metadata):
        """
        Is there at least one sample from each class?
        
        If a class is completely missing:
        - StratifiedGroupKFold throws an error (cannot stratify)
        - The model never sees that class
        
        nunique() → How many distinct values are there?
        """


        unique_targets = metadata['targets'].nunique()
        assert unique_targets == cfg.NUM_CLASSES, \
            f"Expected {cfg.NUM_CLASSES} classes, found {unique_targets}"

    def test_no_duplicate_images(self, metadata):
        """
        Does the same image name appear more than once?
        
        If there are duplicates → the same image could end up in both the train and val sets → data leakage.
        
        duplicated() → returns True/False for each row (does the same value already exist?)
        .sum() → returns the number of True values (number of duplicates)
        """

        duplicates = metadata['image'].duplicated().sum()
        assert duplicates == 0, f"Found {duplicates} duplicate image names"


    def test_lession_id_filled(self, metadata):
        """
        Does `lesion_id` contain any null values?
        
        StratifiedGroupKFold performs grouping based on `lesion_id`.
        If there are null lesion_id values → grouping breaks down → risk of data leakage.
        
        isnull() → returns True/False for each value (is it NaN?)
        .sum() → counts the number of True values (number of nulls)
        """
        null_count = metadata['lesion_id'].isnull().sum()
        assert null_count ==0, f"Found {null_count} null lession_ids"


        
class TestDatasetIntegratiy:
    def test_dataset_folder_exists(self):
        assert os.path.exists(cfg.DATA_PATH), \
            pytest.skip(f"Dataset  folder not found: {cfg.DATA_PATH}")
        
    def test_correct_number_of_class_folders(self):
        """
        Are there the correct number of class folders?
        8 classes = 8 folders.
        If there are 7 folders → one class is missing → the model learns 7 classes
        but there are 8 class outputs → rubbish predictions.
        """

        if not os.path.exists(cfg.DATA_PATH):
            pytest.skip(f"Dataset not found: {cfg.DATA_PATH}")

        folders = [f for f in os.listdir(cfg.DATA_PATH)
                   if os.path.isdir(os.path.join(cfg.DATA_PATH, f))]
        assert len(folders) == cfg.NUM_CLASSES, \
            f"Excepted {cfg.NUM_CLASSES} class folders, found {len(folders)}: {folders}"
        
    def test_class_folders_match_config(self):
        """
        Do the folder names match the CLASSES in the config?

        Config: ['0_mel', '1_nv', '2_bcc', ...]
        Folders: ['0_mel', '1_nv', '2_bcc', ...]

        If the folder is "mel" but the config says "0_mel" → ImageFolder sorts incorrectly
        → labels shift → the model learns rubbish.
        """

        if not os.path.exists(cfg.DATA_PATH):
            pytest.skip(f"Dataset not found: {cfg.DATA_PATH}")

        folders = sorted(os.listdir(cfg.DATA_PATH))
        excepted = sorted(cfg.CLASSES)
        assert folders == excepted , \
            f"Folder mismatch. \nExcepted: {excepted}\nFound: {folders}"
        
    def test_no_empty_class_folders(self):
        """
        Is there an empty class folder?

        If there is an empty folder → ImageFolder skips that class → the number of classes
        decreases → the index mapping breaks.

        It is particularly important to check after running bulk_crop.py — YOLO may delete all files
        in a class where it could not find any lesions.
        """

        if not os.path.exists(cfg.DATA_PATH):
            pytest.skip(f"Dataset not found: {cfg.DATA_PATH}")

        for class_name in cfg.CLASSES:
            class_path = os.path.join(cfg.DATA_PATH, class_name)
            if os.path.exists(class_path):
                files = os.listdir(class_path)
                assert len(files) > 0, f"Empty class folder: {class_name}"



