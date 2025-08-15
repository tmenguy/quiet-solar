import copy


def is_amps_zero(amps: list[float | int]) -> bool:
    if amps is None:
        return True

    for a in amps:
        if a != 0.0:
            return False

    return True


def are_amps_equal(left_amps: list[float | int], right_amps: list[float | int]) -> bool:
    for i in [0,1,2]:
        if left_amps[i] != right_amps[i]:
            return False
    return True


def is_amps_greater(left_amps: list[float | int], right_amps: list[float | int]):
    for i in range(3):
        if left_amps[i] > right_amps[i]:
            return True
    return False


def add_amps(left_amps: list[float | int], right_amps: list[float | int]) -> list[float | int]:
    if left_amps is None and right_amps is None:
        return [0.0, 0.0, 0.0]
    elif left_amps is None:
        return copy.copy(right_amps)
    elif right_amps is None:
        return copy.copy(left_amps)

    adds = [left_amps[i] + right_amps[i] for i in range(3)]
    return adds


def diff_amps(left_amps: list[float | int], right_amps: list[float | int]) -> list[float | int]:
    if left_amps is None or right_amps is None:
        return [0.0, 0.0, 0.0]

    diff = [left_amps[i] - right_amps[i] for i in range(3)]
    return diff


def min_amps(left_amps: list[float | int], right_amps: list[float | int]) -> list[float | int]:
    if left_amps is None and right_amps is None:
        return [0.0, 0.0, 0.0]
    elif left_amps is None:
        return copy.copy(right_amps)
    elif right_amps is None:
        return copy.copy(left_amps)

    mins = [min(left_amps[i], right_amps[i]) for i in range(3)]
    return mins


def max_amps(left_amps: list[float | int], right_amps: list[float | int]) -> list[float | int]:
    if left_amps is None and right_amps is None:
        return [0.0, 0.0, 0.0]
    elif left_amps is None:
        return copy.copy(right_amps)
    elif right_amps is None:
        return copy.copy(left_amps)

    maxs = [max(left_amps[i], right_amps[i]) for i in range(3)]
    return maxs
