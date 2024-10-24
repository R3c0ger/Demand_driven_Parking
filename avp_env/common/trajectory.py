class Trajectory:
    def __init__(self, data):
        if "path_id" in data:
            self.scan = data["scan"]
            self.path_id = data["path_id"]
            self.instruction = data["instruction"]
            self.tags = data["tags"]
            self.ParkingID = data["ParkingID"]
            self.loc_id = data["loc_id"]
        else:
            self.scan = data["scan"]
            self.instruction = data["instruction"]
