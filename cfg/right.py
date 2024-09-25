env_cls = "ICCGANRightHand"
env_params = dict(
    episode_length = 600, 
    character_model = "assets/right_hand_guitar.xml",
    motion_file = "assets/motions/scale.json",
    note_file = [],

    goal_reward_weight=0.5,

    key_links = [
        "RH:wrist",
        "RH:thumb1", "RH:thumb2", "RH:thumb3",
        "RH:index1", "RH:index2", "RH:index3",
        "RH:middle1", "RH:middle2", "RH:middle3",
        "RH:ring1", "RH:ring2", "RH:ring3",
        "RH:pinky1", "RH:pinky2", "RH:pinky3"
    ],
    parent_link = "guitar"
)

training_params = dict(
    max_epochs =    100000,
    save_interval =  50000,
    terminate_reward = -25
)

discriminators = {
    "RH/hand": dict(
        key_links = [
            "RH:wrist", 
            "RH:thumb1", "RH:thumb2", "RH:thumb3",
            "RH:index1", "RH:index2", "RH:index3",
            "RH:middle1", "RH:middle2", "RH:middle3",
            "RH:ring1", "RH:ring2", "RH:ring3",
            "RH:pinky1", "RH:pinky2", "RH:pinky3"
        ],
        parent_link = "guitar",
        motion_file = "assets/right_hand_motions.yaml",
    )
}
