# logging file
log_f = open(log_f_name,"w+")
log_f.write('episode,timestep,reward\n')


# printing and logging variables
print_running_reward1 = 0
print_running_reward2 = 0

print_running_episodes = 0

log_running_reward1 = 0
log_running_reward2 = 0

log_running_episodes = 0

time_step = 0
i_episode = 0


# training loop
while time_step <= max_training_timesteps:
    
    state1 = env.reset_a()
    state2= env.reset_b()

    current_ep_reward1 = 0
    current_ep_reward2 = 0
    for t in range(1, max_ep_len+1):
        
        # select action with policy
        action1 = ppo_agent1.select_action(state1)

        action2=ppo_agent2.select_action(state2)
       
        state1, reward1, done, _ = env.step_a(action1)

        state2, reward2, done, _ = env.step_b(action2)

        # saving reward and is_terminals
        ppo_agent1.buffer.rewards.append(reward1)
        ppo_agent1.buffer.is_terminals.append(done)
        
        ppo_agent2.buffer.rewards.append(reward2)
        ppo_agent2.buffer.is_terminals.append(done)
        
        time_step +=1
        
        current_ep_reward1 += reward1
        current_ep_reward2+=reward2
        # update PPO agent
        if time_step % update_timestep == 0:
            ppo_agent1.update()
            ppo_agent2.update()

        # if continuous action space; then decay action std of ouput action distribution
        if has_continuous_action_space and time_step % action_std_decay_freq == 0:
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

        # log in logging file
        if time_step % log_freq == 0:

            # log average reward till last episode
            log_avg_reward1 = log_running_reward1 / log_running_episodes
            log_avg_reward1 = round(log_avg_reward1, 4)

            log_avg_reward2 = log_running_reward2 / log_running_episodes
            log_avg_reward2 = round(log_avg_reward2, 4)
 
  
            log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward1))
            log_f.flush()
             
            log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward2))
            log_f.flush()

            log_running_reward = 0
            log_running_episodes = 0

        # printing average reward
        if time_step % print_freq == 0:

            # print average reward till last episode
            print_avg_reward1 = print_running_reward1 / print_running_episodes
            print_avg_reward1 = round(print_avg_reward1, 2)

            print_avg_reward2 = print_running_reward2 / print_running_episodes
            print_avg_reward2 = round(print_avg_reward2, 2)

            print("Episode : {} \t\t Timestep : {} \t\t Average Reward1 : {}".format(i_episode, time_step, print_avg_reward1))

            print("Episode : {} \t\t Timestep : {} \t\t Average Reward2 : {}".format(i_episode, time_step, print_avg_reward2))

            print_running_reward = 0
            print_running_episodes = 0
            
        # save model weights
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path1)
            ppo_agent1.save(checkpoint_path1)
            ppo_agent2.save(checkpoint_path2)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")
            
        # break; if the episode is over
        if done:
            break

    print_running_reward1 += current_ep_reward1
    print_running_reward2 += current_ep_reward2
    print_running_episodes += 1

    log_running_reward1 += current_ep_reward1
    log_running_reward2 += current_ep_reward2
    log_running_episodes += 1

    i_episode += 1


log_f.close()
env.close()




# print total training time
print("============================================================================================")
end_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("Finished training at (GMT) : ", end_time)
print("Total training time  : ", end_time - start_time)
print("============================================================================================")

################################ End of Part II ################################

obs1 = env.reset_a()
obs2=env.reset_b()
n_steps = 50
for step in range(n_steps):
  action_a= ppo_agent1.select_action(obs1)
  action_b = ppo_agent2.select_action(obs2)
  print("Step {}".format(step + 1))
  print("Action:a", action_a)
  print("Action:b",action_b)
  obs1, reward1, done1, info = env.step_a(action_a)
  obs2, reward2, done2, info = env.step_b(action_b)
  #print('obs a =', obs1, 'reward a =', reward1, 'done=', done1)
  #print('obs b =', obs2, 'reward b =', reward2, 'done=', done2)
  env.render(mode='console')
  if done1 :
    # Note that the VecEnv resets automatically
    # when a done signal is encountered
    print("Goal reached! by a", "reward_a=", reward1)
    #print("Goal reached!","reward_b=",reward2)
    break
  if done2 :
    # Note that the VecEnv resets automatically
    # when a done signal is encountered
    #print("Goal reached!", "reward_a=", reward1)
    print("Goal reached! by b","reward_b=",reward2)
    break
