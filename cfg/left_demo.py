env_cls = "ICCGANLeftHand"
env_params = dict(
    episode_length = 600,
    character_model = "assets/left_hand_guitar.xml",
    motion_file = "assets/motions/scale.json",
    note_file = 9527,

    random_pitch_rate = 0,
    random_bpm_rate = 0,
    merge_repeated_notes = False,

    goal_reward_weight=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15],

    key_links = [
        "LH:wrist",
        "LH:thumb1", "LH:thumb2", "LH:thumb3",
        "LH:index1", "LH:index2", "LH:index3",
        "LH:middle1", "LH:middle2", "LH:middle3",
        "LH:ring1", "LH:ring2", "LH:ring3",
        "LH:pinky1", "LH:pinky2", "LH:pinky3",
    ],
    parent_link = "guitar"
)

training_params = dict(
    max_epochs =   100000,
    save_interval = 20000,
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
    )
}
