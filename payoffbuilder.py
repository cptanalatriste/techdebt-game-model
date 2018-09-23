import tensorflow as tf
import numpy as np
import logging

from tqdm import tqdm
import itertools

import rlagent
import simmodel
import trainingdriver


def load_heuristic_agents(logger):
    agent_map = {0: simmodel.BaseDeveloper(name="last-minute-patcher", logger=logger, panic_threshold=7,
                                           action_selector=simmodel.last_minute_patcher),
                 1: simmodel.BaseDeveloper(name="stressed-patcher", logger=logger, panic_threshold=2,
                                           action_selector=simmodel.stressed_patcher)}

    return agent_map


def load_trained_agents(logger, scenario_name):
    agent_map = {}
    for agent_index in range(trainingdriver.NUMBER_AGENTS):
        agent = rlagent.DeepQLearner(
            name=trainingdriver.DEVELOPER_NAME_PREFIX + str(agent_index) + "_" + scenario_name,
            input_number=trainingdriver.INPUT_NUMBER,
            hidden_units=trainingdriver.HIDDEN_UNITS,
            logger=logger)

        agent_map[agent_index] = agent

    return agent_map


def main():
    log_filename = "payoff_table_builder.log"
    logging_level = logging.INFO
    logger = logging.getLogger("Payoff-table-builder->")
    handler = logging.FileHandler(log_filename, mode='w')
    logger.addHandler(handler)
    logger.setLevel(logging_level)

    simulation_episodes = 100
    use_rl_agents = False

    # This are the values for the current scenario
    # sloppy_rework_factor = 1.05
    # avg_resolution_time = trainingdriver.SCENARIO_AVG_RESOLUTION_TIME

    # This is the value for the worse scenario
    # sloppy_rework_factor = 1.05
    # avg_resolution_time = 1 / 5.0

    # This is the value for the better scenario
    sloppy_rework_factor = 10
    avg_resolution_time = 1 / 5.0

    scenario_name = "sloppy_code_impact_105"
    checkpoint_path = "results/sloppy_code_impact_105_5000_steps/" + scenario_name + trainingdriver.CHECKPOINT_SUFFIX

    scenario_approach_map = {simmodel.CLEAN_ACTION: trainingdriver.CLEAN_CODING_APPROACH,
                             simmodel.SLOPPY_ACTION: simmodel.CodingApproach(
                                 resolution_factor=trainingdriver.SLOPPY_RESOLUTION_FACTOR,
                                 rework_factor=sloppy_rework_factor,
                                 code_impact=trainingdriver.SLOPPY_CODE_IMPACT)}

    agent_map = load_heuristic_agents(logger)
    if use_rl_agents:
        agent_map = load_trained_agents(logger, scenario_name)
        saver = tf.train.Saver()

    with tf.Session() as session:
        if use_rl_agents:
            saver.restore(session, checkpoint_path)
            logger.info("Restored: " + checkpoint_path)
        else:
            logger.info("Using only heuristic strategies!!!")

        for agent_index, opponent_index in itertools.combinations_with_replacement(range(trainingdriver.NUMBER_AGENTS),
                                                                                   trainingdriver.NUMBER_AGENTS):
            logger.info("Strategy Profile: DEV %d vs DEV %d", agent_index, opponent_index)

            simulation_environment = simmodel.SimulationEnvironment(logger=logger,
                                                                    time_units=trainingdriver.SCENARIO_TIME_UNITS,
                                                                    avg_resolution_time=avg_resolution_time,
                                                                    prob_new_issue=trainingdriver.SCENARIO_PROB_NEW_ISSUE,
                                                                    prob_rework=trainingdriver.SCENARIO_PROB_REWORK)

            developers = [simmodel.Developer(agent=agent_map[agent_index], approach_map=scenario_approach_map),
                          simmodel.Developer(agent=agent_map[opponent_index], approach_map=scenario_approach_map)]

            for episode_index in tqdm(range(1, simulation_episodes + 1)):
                simulation_environment.reset(developers)

                for time_step in range(simulation_environment.time_units):
                    simulation_environment.step(developers=developers,
                                                session=session)

                logger.debug("Strategy Profile: DEV %d vs DEV %d ->Episode %s finished: ", agent_index,
                             opponent_index,
                             str(episode_index))

                for developer in developers:
                    developer.log_progress()

            for developer in developers:
                payoff_values = [performance_metric.issues_delivered for performance_metric in
                                 developer.agent.metric_catalogue]
                sloppiness_values = [performance_metric.get_sloppy_ratio() for performance_metric in
                                     developer.agent.metric_catalogue]

                logger.info(
                    "%s ->  %.2f REPLICATIONS : Payoff (mean, std) %.2f %.2f   Sloppiness (mean, std): %.2f %.2f",
                    developer.name,
                    simulation_episodes, np.mean(payoff_values), np.std(payoff_values), np.mean(sloppiness_values),
                    np.std(sloppiness_values))


if __name__ == "__main__":
    main()
