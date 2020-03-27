from agents.QLearningAgent import QLearningAgent
from environments.EnvironmentLoader import EnvironmentLoader
import pygame
import sys
from settings import RenderSettings, SaveSettings
from agents.AgentManager import AgentManager


class QLearning:

    def __init__(self, epsilon_policy, map_name, hyperparameters=None, save_name=None):
        self.hyperparameters = hyperparameters
        self.epsilon_policy = epsilon_policy
        self.map = map_name
        self.save_name = save_name

    def event_occured(self, timeout_ms: int = 0, renderer=None):
        timer_running = True
        start_ticks = pygame.time.get_ticks()

        while timer_running:

            # check if mouse hover on arrow
            if renderer is not None:
                hover_detected = False
                for area in renderer.hover_areas:
                    points = eval(area)
                    rect = pygame.Rect(points)
                    if rect.collidepoint(pygame.mouse.get_pos()):
                        renderer.update_hover(renderer.hover_areas[area])
                        hover_detected = True
                if not hover_detected:
                    renderer.update_hover(None)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)
                elif event.type == pygame.KEYDOWN and pygame.key.get_pressed()[pygame.K_r]:
                    return "reset"
                elif event.type == pygame.KEYDOWN and pygame.key.get_pressed()[pygame.K_s]:
                    return "skip"
                elif event.type == pygame.KEYDOWN and pygame.key.get_pressed()[pygame.K_p]:
                    return "pause"
            timer_running = (pygame.time.get_ticks() - start_ticks) < timeout_ms
        return None

    def step(self, agent, env) -> bool:
        state = env.get_current_state()
        possible_actions = env.get_possible_actions()
        action = agent.get_action(state, possible_actions)
        next_state, reward, done = env.step(action)
        next_state_possible_actions = env.get_possible_actions()

        print(f"state: {state}, action: {action} -> reward: {reward}")

        agent.update(state, action, reward, next_state, next_state_possible_actions, done)
        return done

    def train(self):
        # load environment
        loader = EnvironmentLoader("environments/maps")
        env = loader.load_map(self.map)

        # set up agent

        agent = QLearningAgent(self.hyperparameters["alpha"],
                               self.epsilon_policy,
                               self.hyperparameters["discount"],
                               env.action_space,
                               env.state_space)

        # reset everything
        epoch_counter = 0
        agent.reset()
        env.reset()

        # generate epsilon values (runs infinitely long)
        for epsilon in self.epsilon_policy:

            # set epsilon for current epoch
            agent.epsilon = epsilon

            env.reset_position()
            done = False

            while not done:
                render_current_epoch = RenderSettings.ENABLED and epoch_counter % RenderSettings.INTERVAL == 0
                save_current_epoch = SaveSettings.ENABLED and epoch_counter % SaveSettings.INTERVAL == 0

                if epoch_counter % RenderSettings.UPDATE_FREQ_TITLE == 0 or render_current_epoch:
                    pygame.display.set_caption(f'KI-Labor GridWorld - Epoch {epoch_counter}')  # takes around 0.1ms on average
                    env.renderer.update_info(self.hyperparameters, epsilon)

                if render_current_epoch:
                    event = self.event_occured(timeout_ms=RenderSettings.TIME_BETWEEN_FRAMES, renderer=env.renderer)
                else:
                    event = self.event_occured(renderer=env.renderer)

                if render_current_epoch:
                    env.render(agent.get_q__values())

                if save_current_epoch and self.save_name is not None:
                    AgentManager.save_agent_state(agent, f"{SaveSettings.SAVE_PATH}/{self.save_name}_{epoch_counter}.txt")

                done = self.step(agent, env)

                if done and render_current_epoch and event is None:
                    event = self.event_occured(timeout_ms=RenderSettings.TIME_BETWEEN_FRAMES, renderer=env.renderer)
                    env.render(agent.get_q__values(), True)

                # handle events
                if event == "reset":
                    return
                elif event == "skip":
                    break
                elif event == "pause":
                    while self.event_occured(renderer=env.renderer) != "pause":
                        continue

            epoch_counter += 1

    def test(self, q_values):
        # load environment
        loader = EnvironmentLoader("environments/maps")
        env = loader.load_map(self.map)

        # set up agent

        agent = QLearningAgent(alpha=0,
                               epsilon_policy=self.epsilon_policy,
                               discount=0,
                               action_space=env.action_space,
                               state_space=env.state_space,
                               q_values=q_values)

        # reset everything
        agent.reset()
        env.reset()

        # generate epsilon values (runs infinitely long)
        for epsilon in self.epsilon_policy:

            # set epsilon for current epoch
            agent.epsilon = epsilon

            env.reset_position()
            done = False

            pygame.display.set_caption(f'KI-Labor GridWorld - Test')

            while not done:
                env.renderer.update_info(self.hyperparameters, epsilon)

                event = self.event_occured(timeout_ms=RenderSettings.TIME_BETWEEN_FRAMES, renderer=env.renderer)
                env.render(q_values)

                done = self.step(agent, env)

                if done and event is None:
                    event = self.event_occured(timeout_ms=RenderSettings.TIME_BETWEEN_FRAMES, renderer=env.renderer)
                    env.render(q_values, True)

                # handle events
                if event == "reset":
                    return
                elif event == "skip":
                    break
                elif event == "pause":
                    while self.event_occured(renderer=env.renderer) != "pause":
                        continue