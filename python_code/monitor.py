import matplotlib.pyplot as plt

class Monitor:
  def __init__(self, version):
    # Plot agent performance
    self.scores, self.human_scores, self.episodes, self.averages = [], [], [], []
    fig, self.ax1 = plt.subplots(1, 1, figsize=(18, 9))
    self.ax1.set_ylabel('Score', fontsize=15)
    self.ax1.set_xlabel('Episode', fontsize=15)

    self.version = version
    
  def PlotModel(self, score, human_score, episode, agent_name):
    window_size = 50
    self.scores.append(score)
    self.human_scores.append(human_score)
    self.episodes.append(episode)        
    if len(self.scores) > window_size:
        # moving avrage:
        self.averages.append(sum(self.scores[-1 * window_size: ]) / window_size)
    else:
        self.averages.append(sum(self.scores) / len(self.scores))

    self.ax1.plot(self.scores, 'b')
    # self.ax1.plot(self.human_scores, 'r')
    self.ax1.plot(self.averages, 'r')

    try:
        plt.savefig("data/images/"+agent_name+"_"+self.version+".png", dpi = 150)
    except OSError:
        pass

    return str(self.averages[-1])[:5]