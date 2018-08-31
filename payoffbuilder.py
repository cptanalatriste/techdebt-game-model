import tensorflow as tf
import logging

from tqdm import tqdm

import rlagent
import simmodel
import trainingdriver


def main():
    log_filename = "payoof_table_builder.log"
    logging_level = logging.INFO
    logger = logging.getLogger("Payoff-table-builder->")
    handler = logging.FileHandler(log_filename, mode='w')
    logger.addHandler(handler)
    logger.setLevel(logging_level)

    simulation_episodes = 30

    developers = []

    scenario_name = "sloppy_code_impact_105"
    checkpoint_path = "results/sloppy_code_impact_105/" + scenario_name + trainingdriver.CHECKPOINT_SUFFIX
    sloppy_rework_factor = 1.05
    scenario_approach_map = {simmodel.CLEAN_ACTION: trainingdriver.CLEAN_CODING_APPROACH,
                             simmodel.SLOPPY_ACTION: simmodel.CodingApproach(
                                 resolution_factor=trainingdriver.SLOPPY_RESOLUTION_FACTOR,
                                 rework_factor=sloppy_rework_factor,
                                 code_impact=trainingdriver.SLOPPY_CODE_IMPACT)}

    for agent_index in range(trainingdriver.NUMBER_AGENTS):
        developer_agent = rlagent.DeepQLearner(
            name=trainingdriver.DEVELOPER_NAME_PREFIX + str(agent_index) + "_" + scenario_name,
            input_number=trainingdriver.INPUT_NUMBER,
            hidden_units=trainingdriver.HIDDEN_UNITS,
            logger=logger, learning_rate=None)
        developer = simmodel.Developer(agent=developer_agent, approach_map=scenario_approach_map)
        developers.append(developer)

    saver = tf.train.Saver()

    with tf.Session() as session:
        saver.restore(session, checkpoint_path)
        logger.info("Restored: " + checkpoint_path)

        simulation_environment = simmodel.SimulationEnvironment(logger=logger,
                                                                time_units=trainingdriver.SCENARIO_TIME_UNITS,
                                                                avg_resolution_time=trainingdriver.SCENARIO_AVG_RESOLUTION_TIME,
                                                                prob_new_issue=trainingdriver.SCENARIO_PROB_NEW_ISSUE,
                                                                prob_rework=trainingdriver.SCENARIO_PROB_REWORK)

        for episode_index in tqdm(range(1, simulation_episodes + 1)):
            simulation_environment.reset(developers)

            for time_step in range(simulation_environment.time_units):
                simulation_environment.step(developers=developers, time_step=time_step,
                                            session=session)

            logger.debug("Episode %s finished: ", str(episode_index))


if __name__ == "__main__":
    main()
