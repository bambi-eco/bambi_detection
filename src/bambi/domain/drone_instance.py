
from bambi.domain.camera import Camera
from bambi.domain.manufacturer import Manufacturer
from bambi.domain.drone import Drone
from typing import Optional, List, Union



# ADD NEW DRONES HERE
name_to_serial = {                                 # internal folder name
    "M30T_01"            : "1581F62HD226U00B7033", # M30T_33
    "M30T_02"            : "1581F62HD226100BV060", # M30T_60
    "M3T_Spektakulair"   : "1581F5FJB22A700A0DW3", # M3TE_W3
    "flying dutchman 2"  : "1581F5FJB22A700A0DV7",  # M3TE_V7
    "M30T_VC"            : "FFFF",  # TODO: replace with actual SN
    "M30T_ASP"           : "FFFE",
    "M3T_02_Spektakulair": "1581F5FJD23BD00EQCW0",
}

name_to_type = {
    "M30T_01"            : Drone.M30T,
    "M30T_02"            : Drone.M30T,
    "M3T_Spektakulair"   : Drone.M3T,
    "flying dutchman 2"  : Drone.M3T,
    "M30T_VC"            : Drone.M30T,
    "M30T_ASP"           : Drone.M30T,
    "M3T_02_Spektakulair": Drone.M3T
}

name_to_owner = {
    "M30T_01"            : "FH",
    "M30T_02"            : "FH",
    "M3T_Spektakulair"   : "Spektakulair",
    "flying dutchman 2"  : "BW",
    "M30T_VC"            : "VC",
    "M30T_ASP"           : "Land OOE",
    "M3T_02_Spektakulair": "Spektakulair",
}

serial_to_name = {v: k for k, v in name_to_serial.items()}
serial_to_type = {serial: name_to_type[serial_to_name[serial]] for serial in serial_to_name}
serial_to_owner = {serial: name_to_owner[serial_to_name[serial]] for serial in serial_to_name}


class DroneInstance:
    def __init__(
        self,
        internal_name: str,
        serial_id: str = "",
        owner: str = "",
        type: Optional[Union[str, Drone]] = None,
    ):
       
        self.internal_name = internal_name
        
        if serial_id == "":
            if internal_name not in name_to_serial:
                raise Exception(f"Drone name {internal_name} not found!")
            self.serial_id = name_to_serial[internal_name]
        else:
            self.serial_id = serial_id
            
        if type is None:
            if internal_name not in name_to_type:
                raise Exception(f"Drone name {internal_name} not found!")
            type = name_to_type[internal_name]    
        self.type = Drone(type)
        
        if owner == "":
            if internal_name not in name_to_owner:
                raise Exception(f"Drone name {internal_name} not found!")
            owner = name_to_owner[internal_name]
        self.owner = owner
        
    def __str__(self) -> str:
        """Returns a string representation of the drone instance: a combination of type and the last two digits of the serial number."""
        return self.get_unique_name()

    def get_unique_name(self) -> str:
        """Returns a string representation of the drone instance: a combination of type and the last two digits of the serial number."""
        return self.type.to_4str() + "_" + self.serial_id[-2:]
    
    def to_unique_name(self) -> str:
        """Returns a string representation of the drone instance: a combination of type and the last two digits of the serial number."""
        return self.get_unique_name()
    
    @staticmethod
    def from_internal_name(name: str) -> "DroneInstance":
        """Returns a drone instance from an internal name."""
        return DroneInstance(name)
    
    @staticmethod
    def from_unique_name(name: str) -> "DroneInstance":
        """Returns a drone instance from a unique name."""
        for d in get_available_drones():
            if d.get_unique_name() == name:
                return d
        raise Exception(f"Drone name {name} not found!")


def get_available_drones() -> List[DroneInstance]:
    """Returns a list of all drone instances."""
    return [DroneInstance.from_internal_name(k) for k in name_to_serial.keys()]

# EXAMPLE USAGE
# if __name__ == "__main__":

#     for n in name_to_serial.keys():
#         d = DroneInstance(n)
#         print(d)