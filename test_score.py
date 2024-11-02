import json

with open('test_results.json', 'r') as file:
    context_data = json.load(file)


# 开始计算


def get_parking_metrics(experiments):
    NE_metrics = calculate_navigation_errors(experiments)
    SR_metrics = calculate_success_rate(experiments)
    DWSR_metrics = calculate_weighted_success_rate(experiments)
    APE_metrics = calculate_absolute_parking_slot_error(experiments)
    MR_metrics = calculate_miss_rate(experiments)
    PSMD_metrics = calculate_matching_rate(experiments)
    return NE_metrics, SR_metrics, DWSR_metrics, APE_metrics, MR_metrics, PSMD_metrics


def calculate_navigation_errors(experiments):
    # 初始化错误次数
    error_count = 0
    # 总实验次数
    total_experiments = len(experiments)

    # 遍历每次实验的结果
    for experiment in experiments:
        if experiment["target_features"]:
            result_tags = experiment["result_features"]['tags']
            if result_tags["Occupied"] != 0:
                error_count += 1

    # 计算导航错误率
    error_rate = error_count / total_experiments

    # 返回导航错误率（百分比形式）
    return error_rate


def calculate_success_rate(experiments):
    # 初始化成功次数
    success_count = 0
    error_count = 0
    # 总实验次数
    total_experiments = len(experiments)
    # 遍历每次实验的结果
    for experiment in experiments:
        result_id = experiment["result_id"]
        target_id = experiment["target_id"]
        # 检查系统决策的停车位是否与最佳停车位匹配
        if result_id and result_id in target_id:
            success_count += 1
        else:
            error_count += 1

    # 计算成功率
    success_rate = success_count / total_experiments

    # 返回成功率（百分比形式）
    return success_rate


def calculate_weighted_success_rate(experiments):
    """
       计算距离加权的停车系统成功率

       参数:
       experiments (list of dict): 每个字典包含 result_id, target_id 和 distance，
                                   例如:
                                   [
                                       {"result_id": 5, "target_id": 5, "distance": 10},
                                       {"result_id": 12, "target_id": 8, "distance": 15},
                                       ...
                                   ]
       返回:
       float: 距离加权成功率（以百分比表示）
    """
    # 初始化成功次数
    weighted_success_sum = 0
    # 总实验次数
    total_experiments = len(experiments)
    # 遍历每次实验的结果
    for experiment in experiments:
        result_id = experiment["result_id"]
        target_id = experiment["target_id"]
        distance = experiment["result_features"]["distance"]

        # 检查系统决策的停车位是否与最佳停车位匹配
        if result_id and result_id in target_id:
            weighted_success_sum += 1 / distance

    # 计算成功率
    weighted_success_rate = weighted_success_sum / total_experiments

    # 返回成功率（百分比形式）
    return weighted_success_rate


def calculate_absolute_parking_slot_error(experiments):
    # 初始化误差和实验次数
    total_error = 0
    total_experiments = len(experiments)

    min_distance = 0
    # 遍历每次实验的结果
    for experiment in experiments:
        target_positions = [target["distance"] for target in experiment["target_features"]]
        result_position = experiment["result_features"]["distance"]

        # 跳过没有决策车位或没有理想车位的实验
        if not result_position and target_positions:
            min_distance = min(abs((87 - target_position) / 87) for target_position in target_positions)
        elif not target_positions and result_position:
            min_distance = min(abs((result_position - 87) / 87), abs(result_position / 87))
        elif not target_positions and not result_position:
            min_distance = 0
        elif target_positions and result_position:
            # 计算决策车位与每个理想车位的绝对距离，选择最近的距离
            min_distance = min(abs((result_position - target_position) / 87) for target_position in target_positions)

        # 累加最小距离
        total_error += min_distance

    # 计算绝对车位误差
    ape = total_error / total_experiments

    return ape


def calculate_miss_rate(experiments):
    # 初始化错失总数和总指令数
    total_miss = 0
    total_experiments = 0

    # 遍历每次实验的结果
    for experiment in experiments:

        target_positions = [target["distance"] for target in experiment["target_features"]]
        result_position = experiment["result_features"]["distance"]

        if target_positions and result_position:
            targets_before_result = sum(1 for target_position in target_positions if target_position < result_position)
            # equivalent_miss = targets_before_result / len(target_positions)
            equivalent_miss = targets_before_result / len(target_positions)
            # 累加等效错失车位
            total_miss += equivalent_miss
        total_experiments += 1

    # 计算错失率
    miss_rate = total_miss / total_experiments if total_experiments > 0 else 0

    return miss_rate


def calculate_matching_rate(experiments):
    """
    计算车位匹配度。

    参数:
    experiments (list of dict): 每个字典包含 result_id, target_ids, target_features，
                                例如:
                                [
                                    {
                                        "result_id": (path_id1, loc_id1),
                                        "target_ids": [(path_id2, loc_id2), ...],
                                        "target_features": {"tag": {"tag1": value1, ...}},
                                        "result_tags": {"tag1": value1, ...}
                                    }, ...
                                ]

    返回:
    float: 车位匹配度
    """
    total_matching_score = 0
    total_experiments = len(experiments)

    # 遍历每次实验的结果
    for experiment in experiments:
        if experiment["target_features"]:

            result_id = experiment["result_id"]

            if not result_id:
                continue
            target_tags = experiment["target_features"][0]['tags']
            result_tags = experiment["result_features"]['tags']
            # 跳过没有决策车位或没有用户指令标签的实验

            # 计算完全相同的tags数量
            matching_tags = 0
            for tag, value in target_tags.items():
                if result_tags.get(tag) == value:
                    matching_tags += 1

            # 用户指令包含的tags数量
            total_tags = len(target_tags)

            # 计算匹配度
            matching_score = matching_tags / total_tags if total_tags > 0 else 0

            # 累加匹配度
            total_matching_score += matching_score

    # 计算加权匹配度
    weighted_matching_rate = total_matching_score / total_experiments if total_experiments > 0 else 0

    return weighted_matching_rate


def find_metric_data(test_scenario_id, test_instruction_id, vlp_decision_position_id):
    # result_id 就是 vlp_decision_position_id
    result_id = vlp_decision_position_id

    # 根据 test_scenario_id 和 test_instruction_id 获取 target_id 和 target_features
    target_id = get_target_id(test_scenario_id, test_instruction_id)

    # 根据 test_scenario_id 和 vlp_decision_position_id 获取 result_features
    result_features = get_features_byid(test_scenario_id, vlp_decision_position_id)

    target_features = get_features_bytarid(test_scenario_id, target_id)

    # 构建输出字典
    experiment = {
        "result_id": result_id,
        "target_id": target_id,
        "target_features": target_features,
        "result_features": result_features,
    }

    return experiment


def get_features_byid(test_scenario_id, parking_id):
    features = {}
    if parking_id:
        with open(f'./data/Vision/{test_scenario_id}/parking_slots.json', 'r') as f:
            parking_slots = json.load(f)

        for item in parking_slots:
            if item['ParkingID'] == parking_id:
                features = {
                    "distance": (item['PathID'] - 1) * 3 + (item['LocID'] - 1) % 3 + 1,
                    "tags": {
                        "NextWall": item['NextWall'],
                        "SideRoad": item['SideRoad'],
                        "NearExit": item['NearExit'],
                        "Sunlight": item['Sunlight'],
                        "Column": item['Column'],
                        "NextDriveWay": item['NextDriveWay'],
                        "Charging": item['Charging'],
                        "Disabled": item['Disabled'],
                        "Occupied": item['Occupied'],
                        "Around": item['Around'],
                    },
                }
                break
    else:
        features = {
            "distance": 87,
            "tags": None
        }
    return features


def get_features_bytarid(test_scenario_id, parking_id):
    # 这里需要修改为正确的测试集的内容

    features = []
    for pid in parking_id:
        feature = get_features_byid(test_scenario_id, pid)
        features.append(feature)
    return features


def meets_criteria(slot, tags):
    for key, value in tags.items():
        if slot.get(key) != value:
            return False
    return True


# 示例辅助函数，用于获取 target_id 和 target_features
def get_target_id(test_scenario_id, test_instruction_id):
    with open(f'./data/Vision/{test_scenario_id}/parking_slots.json', 'r') as f:
        parking_slots = json.load(f)

    with open(f'./data/Vision/{test_scenario_id}/Traj.json', 'r') as f:
        parking_commands = json.load(f)

    with open(f'./data/Vision/{test_scenario_id}/Traj_withinfo.json', 'r') as f:
        parking_commands_withinfo = json.load(f)

    test_instruction = parking_commands[test_instruction_id]['instruction']
    match_tags = None
    for item in parking_commands_withinfo:
        if item['instruction'] == test_instruction:
            match_tags = item['tags']
            break
    if match_tags:
        matching_slots = [slot['ParkingID'] for slot in parking_slots if meets_criteria(slot, match_tags)]
    else:
        matching_slots = []
    return matching_slots


if __name__ == '__main__':
    experiments = []
    # 获取传递的信息
    for item in context_data:
        test_scenario_id = item.get("TestScenarioID")
        test_instruction_id = item.get("TestInstructionID")
        vlp_decision_position_id = item.get("VLPDecisionPositionID")

        experiment = find_metric_data(test_scenario_id, test_instruction_id, vlp_decision_position_id)
        experiments.append(experiment)

    # 分别计算
    NE_Metrics, SR_Metrics, DWSR_Metrics, APE_Metrics, MR_Metrics, PSMD_Metrics = get_parking_metrics(experiments)
    # 加权计算（权值可调整）
    mse_result = (
        -10 * NE_Metrics
        + 50 * SR_Metrics
        + 100 * DWSR_Metrics
        - 10 * APE_Metrics
        - 5 * MR_Metrics
        + 40 * PSMD_Metrics
    )
    print(mse_result)
