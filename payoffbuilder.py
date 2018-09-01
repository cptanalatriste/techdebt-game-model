import tensorflow as tf
import numpy as np
import logging

from tqdm import tqdm
import itertools

import rlagent
import simmodel
import trainingdriver


def main():
    log_filename = "payoff_table_builder.log"
    logging_level = logging.INFO
    logger = logging.getLogger("Payoff-table-builder->")
    handler = logging.FileHandler(log_filename, mode='w')
    logger.addHandler(handler)
    logger.setLevel(logging_level)

    simulation_episodes = 30

    for agent_index, oponent_index in itertools.combinations_with_replacement(range(trainingdriver.NUMBER_AGENTS),
                                                                              trainingdriver.NUMBER_AGENTS):
        logger.info("Strategy Profile: DEV %d vs DEV %d")

        sloppy_rework_factor = 1.05
        scenario_name = "sloppy_code_impact_105"
        checkpoint_path = "results/sloppy_code_impact_105/" + scenario_name + trainingdriver.CHECKPOINT_SUFFIX

        simulation_environment = simmodel.SimulationEnvironment(logger=logger,
                                                                time_units=trainingdriver.SCENARIO_TIME_UNITS,
                                                                avg_resolution_time=trainingdriver.SCENARIO_AVG_RESOLUTION_TIME,
                                                                prob_new_issue=trainingdriver.SCENARIO_PROB_NEW_ISSUE,
                                                                prob_rework=trainingdriver.SCENARIO_PROB_REWORK)

        scenario_approach_map = {simmodel.CLEAN_ACTION: trainingdriver.CLEAN_CODING_APPROACH,
                                 simmodel.SLOPPY_ACTION: simmodel.CodingApproach(
                                     resolution_factor=trainingdriver.SLOPPY_RESOLUTION_FACTOR,
                                     rework_factor=sloppy_rework_factor,
                                     code_impact=trainingdriver.SLOPPY_CODE_IMPACT)}

        developers = []

        agent = rlagent.DeepQLearner(
            name=trainingdriver.DEVELOPER_NAME_PREFIX + str(agent_index) + "_" + scenario_name,
            input_number=trainingdriver.INPUT_NUMBER,
            hidden_units=trainingdriver.HIDDEN_UNITS,
            logger=logger)
        developers.append(simmodel.Developer(agent=agent, approach_map=scenario_approach_map))

        opponent = rlagent.DeepQLearner(
            name=trainingdriver.DEVELOPER_NAME_PREFIX + str(oponent_index) + "_" + scenario_name,
            input_number=trainingdriver.INPUT_NUMBER,
            hidden_units=trainingdriver.HIDDEN_UNITS,
            logger=logger)
        developers.append(simmodel.Developer(agent=opponent, approach_map=scenario_approach_map))



        saver = tf.train.Saver()

        with tf.Session() as session:
            saver.restore(session, checkpoint_path)
            logger.info("Strategy Profile: DEV %d vs DEV %d ->Restored: " + checkpoint_path, agent_index, oponent_index)

            for episode_index in tqdm(range(1, simulation_episodes + 1)):
                simulation_environment.reset(developers)

                for time_step in range(simulation_environment.time_units):
                    simulation_environment.step(developers=developers, time_step=time_step,
                                                session=session)

                logger.debug("Strategy Profile: DEV %d vs DEV %d ->Episode %s finished: ", agent_index, oponent_index,
                             str(episode_index))

                for developer in developers:
                    developer.log_progress(episode_index)

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
