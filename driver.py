import simmodel
import rlagent
import logging
import dqlearning


def main():
    logging_level = logging.INFO
    # logging_level = logging.DEBUG

    logging.basicConfig(level=logging_level, filename='tech_debt_rl.log', filemode='w')
    logger = logging.getLogger("DQNetwork-Training->")
    logger.debug("Starting script")

    # total_episodes = 1000
    total_episodes = 50
    decay_steps = total_episodes * 10

    train_frequency = 4
    batch_size = 32

    discount_factor = 0.99
    learning_rate = 1e-4
    input_number = 4
    hidden_units = 24

    # time_units = 60
    time_units = 30
    counter_for_learning = total_episodes * time_units * 0.1
    transfer_frequency = counter_for_learning
    save_frequency = counter_for_learning * 0.1

    avg_resolution_time = 1 / 3.0
    prob_new_issue = 0.9
    prob_rework = 0.05
    initial_epsilon = 1.0
    final_epsilon = 0.1

    replay_memory_size = 100
    checkpoint_path = "./tech_debt_rl.ckpt"
    enable_restore = False
    number_agents = 2

    approach_map = {simmodel.CLEAN_ACTION: simmodel.CodingApproach(resolution_factor=1.1, rework_factor=0.9,
                                                                   code_impact=1.0),
                    simmodel.SLOPPY_ACTION: simmodel.CodingApproach(resolution_factor=0.75,
                                                                    rework_factor=1.05,  # TODO Testing rework impact
                                                                    code_impact=1.05)}

    developers = []
    for index in range(number_agents):
        dev_name = "DEV" + str(index)
        developer_agent = rlagent.DeepQLearner(name=dev_name,
                                               learning_rate=learning_rate,
                                               discount_factor=discount_factor,
                                               counter_for_learning=counter_for_learning,
                                               input_number=input_number, hidden_units=hidden_units,
                                               logger=logger,
                                               initial_epsilon=initial_epsilon, final_epsilon=final_epsilon,
                                               decay_steps=decay_steps,
                                               replay_memory_size=replay_memory_size)

        developer = simmodel.Developer(agent=developer_agent, approach_map=approach_map)
        developers.append(developer)

    simulation_environment = simmodel.SimulationEnvironment(logger=logger,
                                                            time_units=time_units,
                                                            avg_resolution_time=avg_resolution_time,
                                                            prob_new_issue=prob_new_issue,
                                                            prob_rework=prob_rework)

    dq_learner = dqlearning.DeepQLearning(logger=logger, total_episodes=total_episodes, decay_steps=decay_steps,
                                          train_frequency=train_frequency, batch_size=batch_size,
                                          counter_for_learning=counter_for_learning,
                                          transfer_frequency=transfer_frequency, save_frequency=save_frequency,
                                          checkpoint_path=checkpoint_path)

    dq_learner.start(simulation_environment, developers, enable_restore)


if __name__ == "__main__":
    main()
