import simmodel
import rlagent
import logging
import dqlearning
import matplotlib.pyplot as plt

DEVELOPER_NAME_PREFIX = "DEV"
INPUT_NUMBER = 4
HIDDEN_UNITS = 24
CHECKPOINT_SUFFIX = "_tech_debt_rl.ckpt"

#Eclipse Platform parameters
SCENARIO_TIME_UNITS = 360
SCENARIO_AVG_RESOLUTION_TIME = 1 / 29.79
SCENARIO_PROB_REWORK = 0.069

NUMBER_AGENTS = 2
SCENARIO_PROB_NEW_ISSUE = 1.0

CLEAN_CODING_APPROACH = simmodel.CodingApproach(
    resolution_factor=0.9,
    rework_factor=0.9,
    code_impact=1.0)

SLOPPY_CODE_IMPACT = 0.95
SLOPPY_RESOLUTION_FACTOR = 1.25


def plot_learning(developers, metric_name, filename="plot.png"):
    print("Plotting before exiting...")
    plt.clf()
    for developer in developers:
        metrics = developer.agent.metric_catalogue
        episodes = range(len(metrics))

        metric_values = [getattr(metric, metric_name) for metric in metrics]
        plt.plot(episodes, metric_values, label=developer.name)

    plt.legend()
    plt.savefig(filename)
    print("Plot stored in " + filename)


def main():
    logging_mode = 'w'
    enable_restore = False

    logging_level = logging.DEBUG

    total_training_steps = 10000
    decay_steps = int(total_training_steps / 2)

    train_frequency = 4
    batch_size = 32

    discount_factor = 0.99
    learning_rate = 1e-4

    counter_for_learning = int(total_training_steps / 100)
    transfer_frequency = counter_for_learning
    save_frequency = counter_for_learning * 0.1

    initial_epsilon = 1.0
    final_epsilon = 0.1

    replay_memory_size = 100

    for sloppy_rework_factor in [1.05]:  # Testing rework impact

        scenario = "sloppy_code_impact_" + str(sloppy_rework_factor).replace('.', '')
        print("Current scenario: " + scenario)

        plot_filename = scenario + '_plot.png'
        log_filename = scenario + '_tech_debt_rl.log'
        checkpoint_path = "./chk/" + scenario + CHECKPOINT_SUFFIX

        logger = logging.getLogger(scenario + "-DQNetwork-Training->")
        handler = logging.FileHandler(log_filename, mode=logging_mode)
        logger.addHandler(handler)
        logger.setLevel(logging_level)

        approach_map = {simmodel.CLEAN_ACTION: CLEAN_CODING_APPROACH,
                        simmodel.SLOPPY_ACTION: simmodel.CodingApproach(resolution_factor=SLOPPY_RESOLUTION_FACTOR,
                                                                        rework_factor=sloppy_rework_factor,
                                                                        code_impact=SLOPPY_CODE_IMPACT)}

        dq_learner = dqlearning.DeepQLearning(logger=logger, total_training_steps=total_training_steps,
                                              decay_steps=decay_steps,
                                              train_frequency=train_frequency, batch_size=batch_size,
                                              counter_for_learning=counter_for_learning,
                                              transfer_frequency=transfer_frequency, save_frequency=save_frequency,
                                              checkpoint_path=checkpoint_path)

        developers = []
        for index in range(NUMBER_AGENTS):
            dev_name = DEVELOPER_NAME_PREFIX + str(index) + "_" + scenario
            developer_agent = rlagent.DeepQLearner(name=dev_name,
                                                   learning_rate=learning_rate,
                                                   discount_factor=discount_factor,
                                                   counter_for_learning=counter_for_learning,
                                                   input_number=INPUT_NUMBER, hidden_units=HIDDEN_UNITS,
                                                   logger=logger,
                                                   initial_epsilon=initial_epsilon, final_epsilon=final_epsilon,
                                                   decay_steps=decay_steps,
                                                   replay_memory_size=replay_memory_size,
                                                   global_step=dq_learner.training_step_var)

            developer = simmodel.Developer(agent=developer_agent, approach_map=approach_map)
            developers.append(developer)

        simulation_environment = simmodel.SimulationEnvironment(logger=logger,
                                                                time_units=SCENARIO_TIME_UNITS,
                                                                avg_resolution_time=SCENARIO_AVG_RESOLUTION_TIME,
                                                                prob_new_issue=SCENARIO_PROB_NEW_ISSUE,
                                                                prob_rework=SCENARIO_PROB_REWORK)

        dq_learner.start(simulation_environment, developers, enable_restore)
        plot_learning(developers, metric_name='issues_delivered', filename=plot_filename)
        plot_learning(developers, metric_name='sloppy_counter', filename='sloppy_counter_' + plot_filename)


if __name__ == "__main__":
    main()
