


# VARIABLE NAME TRACKER TO ASSING MODEL IDENTIFIERS WITH STRINGS (FOR NN LABELING)
class VariableNameTracker:
    def __init__(self):
        self.name_to_obj = {}

    def register(self, name, obj):
        self.name_to_obj[name] = obj

    def get_name(self, obj):
        for name, registered_obj in self.name_to_obj.items():
            if registered_obj is obj:
                return name
        return None