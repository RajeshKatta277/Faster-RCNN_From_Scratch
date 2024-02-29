class CoordinateConverter:
    def __init__(self, image_size=(224, 224), feature_map_size=(7, 7)):
        self.image_size = image_size
        self.feature_map_size = feature_map_size

    def image_to_feature(self, x1, y1, x2, y2):
        """
        Converts the coordinates from image space to feature space.

        Inputs:
            x1, y1, x2, y2 (int): Coordinates of the bounding box in image space.

        Output:
            List: Coordinates of the given bounding box in feature space.
        """
        image_width, image_height = self.image_size
        feature_width, feature_height = self.feature_map_size

        stride_x = image_width / (feature_width)
        stride_y = image_height / (feature_height)

        feature_x1 = x1 / stride_x
        feature_y1 = y1 / stride_y
        feature_x2 = x2 / stride_x
        feature_y2 = y2 / stride_y

        return [feature_x1, feature_y1, feature_x2, feature_y2]

    def feature_to_image(self, feature_x1, feature_y1, feature_x2, feature_y2):
        """
        Converts the coordinates from feature space to image space.

        Inputs:
            feature_x1, feature_y1, feature_x2, feature_y2 (int): Coordinates of the bounding box in feature space.

        Output:
            List: Coordinates of the given bounding box in image space.
        """
        image_width, image_height = self.image_size
        feature_width, feature_height = self.feature_map_size

        stride_x = image_width / (feature_width )
        stride_y = image_height / (feature_height)

        x1 = feature_x1 * stride_x
        y1 = feature_y1 * stride_y
        x2 = feature_x2 * stride_x
        y2 = feature_y2 * stride_y

        return [x1, y1, x2, y2]
