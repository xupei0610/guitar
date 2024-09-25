env_cls = "ICCGANTwoHands"
env_params = dict(
    episode_length = 1200,
    motion_file = "assets/motions/scale.json",
    character_model = ["assets/left_hand_guitar.xml", "assets/right_hand.xml"],
    note_file = 9527,

    random_pitch_rate = 0,
    random_bpm_rate = 0,
    merge_repeated_notes = False,
    
    goal_reward_weight=[0.15/2, 0.15/2, 0.15/2, 0.15/2, 0.15/2, 0.15/2, 1/2],

    key_links = [
        "LH:wrist",
        "LH:thumb1", "LH:thumb2", "LH:thumb3",
        "LH:index1", "LH:index2", "LH:index3",
        "LH:middle1", "LH:middle2", "LH:middle3",
        "LH:ring1", "LH:ring2", "LH:ring3",
        "LH:pinky1", "LH:pinky2", "LH:pinky3",

        "RH:wrist",
        "RH:thumb1", "RH:thumb2", "RH:thumb3",
        "RH:index1", "RH:index2", "RH:index3",
        "RH:middle1", "RH:middle2", "RH:middle3",
        "RH:ring1", "RH:ring2", "RH:ring3",
        "RH:pinky1", "RH:pinky2", "RH:pinky3",
    ],
    parent_link = "guitar",
)

training_params = dict(
    max_epochs =     60000,
    save_interval =  20000,
    terminate_reward = -25
)

discriminators = {
    "LH/wrist": dict(
        key_links = [
            "LH:wrist",
            "LH:thumb1", "LH:thumb2", "LH:thumb3",
        ],
        parent_link = "guitar",
        motion_file = "assets/left_hand_motions.yaml",
        weight = 0.05
    ),
    "LH/fingers": dict(
        key_links = [
            "LH:index1", "LH:index2", "LH:index3",
            "LH:middle1", "LH:middle2", "LH:middle3",
            "LH:ring1", "LH:ring2", "LH:ring3",
            "LH:pinky1", "LH:pinky2", "LH:pinky3",
        ],
        parent_link = "LH:wrist",
        motion_file = "assets/left_hand_motions.yaml",
        weight = 0.05
    ),
    # "RH/hand": dict(
    #     key_links = [
    #         "RH:wrist", 
    #         "RH:thumb1", "RH:thumb2", "RH:thumb3",
    #         "RH:index1", "RH:index2", "RH:index3",
    #         "RH:middle1", "RH:middle2", "RH:middle3",
    #         "RH:ring1", "RH:ring2", "RH:ring3",
    #         "RH:pinky1", "RH:pinky2", "RH:pinky3"
    #     ],
    #     parent_link = "guitar",
    #     motion_file = "assets/right_hand_motions.yaml",
    # )
}
