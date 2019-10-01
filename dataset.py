'''
    Dataset base class
'''

class Dataset(object):
    def __init__(self):
        self._image_ids = []
        self.image_info = []

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    @property
    def image_ids(self):
        return self._image_ids


    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_location(self, image_id):
        info = self.image_info[image_id]
        return info["location"]

    def load_keypoints(self, image_id):
        info = self.image_info[image_id]
        return info["keypoints"]

    def load_quaternion(self, image_id):
        info = self.image_info[image_id]
        return info["quaternion"]

    def load_euler_angles(self, image_id):
        info = self.image_info[image_id]
        return info["pyr"]

    def load_angle_axis(self, image_id):
        info = self.image_info[image_id]
        return info["angleaxis"]

    def load_location_encoded(self, image_id):
        info = self.image_info[image_id]
        return info["location_map"]

    def load_orientation_encoded(self, image_id):
        info = self.image_info[image_id]
        return info["ori_map"]