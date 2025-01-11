def record_video():
    height, width = agent.images[0].shape[0], agent.images[0].shape[1]
    video = cv2.VideoWriter('dqn2048.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            4,
                            (width, height))
    for img in agent.images:
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    video.release()
