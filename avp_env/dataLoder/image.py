import os
import cv2
from avp_env.dataLoder.path import PathLoader


class ImageLoader:
    def __init__(self, env_type, image_shape):
        self.path_loader = PathLoader(env_type)
        self.experiment_paths = self.path_loader.load_path()
        self.image_shape = image_shape
        self.image_data = self._load_images()
        self.render_image = self._load_render_images()

    def _load_images(self):
        image_data = {}
        for experiment_path in self.experiment_paths:
            experiment_id = os.path.basename(experiment_path)
            for filename in os.listdir(experiment_path):
                if filename.endswith('.JPG'):
                    filepath = os.path.join(experiment_path, filename)
                    image_array = cv2.imread(filepath)
                    resized_image = cv2.resize(image_array, (self.image_shape[1], self.image_shape[0]),
                                               interpolation=cv2.INTER_AREA)
                    # image_data[filename] = resized_image

                    # Use a unique key combining experiment_id and filename
                    unique_key = f"{experiment_id}/{filename}"
                    image_data[unique_key] = resized_image
        return image_data

    def _load_render_images(self):
        render_image = {}
        for experiment_path in self.experiment_paths:
            experiment_id = os.path.basename(experiment_path)
            for filename in os.listdir(experiment_path):
                if filename.endswith('.JPG'):
                    filepath = os.path.join(experiment_path, filename)
                    image_array = cv2.imread(filepath)
                    # render_image[filename] = image_array

                    # Use a unique key combining experiment_id and filename
                    unique_key = f"{experiment_id}/{filename}"
                    render_image[unique_key] = image_array

        return render_image
