Hyper Params:
"SEEDC": 5,
    "SEED" : 1,
    "BUFFER_SIZE": 100000, 
    "BATCH_SIZE": 512,
    "GAMMA": 0.998789,
    "TAU": 0.0013,
    "LR_CRITIC": 0.00025,
    "LR_ACTOR": 0.0001,
    "MU":0,
    "THETA": 0.193,
    "SIGMA" : 0.29,
    "EXPLORE" :0.09024
    
  [1] Base Weights files used to achieve Agents scores 0.5 avg 
  #Load trained model weights where got 0.5 in 1800 episodes itself
# Max hit 2.6 and avg 0.3
actornnk_path ="./avgp27Max2p5/checkpoint_actor_point_base_avg2p78.pth"
criticnn_path ="./avgp27Max2p5/checkpoint_critic_point_base_avg2p78.pth"
act_tar_path = "./avgp27Max2p5/checkpoint_actor_target_base_avg2p78.pth"
critc_tar_path="./avgp27Max2p5/checkpoint_critic_target_base_avg2p78.pth"
colabAI.load_trained_model(actornnk_path,criticnn_path,act_tar_path,critc_tar_path)


[2] Saved the the above run (300 Episodes but target achieved in 155th Episodes]
colabAI3 = DDPGMultiAgent(state_size,action_size,replayMemory,num_agents,config)
actornnk_path ="./EnvSolved/checkpoint_actor_point_base_9Sep.pth"
criticnn_path ="./EnvSolved/checkpoint_critic_point_base_9Sep.pth"
act_tar_path = "./EnvSolved/checkpoint_actor_target_base_9Sep.pth"
critc_tar_path="./EnvSolved/checkpoint_critic_target_base_9Sep.pth"
colabAI3.load_trained_model(actornnk_path,criticnn_path,act_tar_path,critc_tar_path)
