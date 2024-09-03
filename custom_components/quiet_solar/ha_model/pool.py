from ..ha_model.on_off_duration import QSOnOffDuration


class QSPool(QSOnOffDuration):


    def support_green_only_switch(self) -> bool:
        return True


    pass