def judge_0(judge_window):
    return f"""You are about to judge a conversation between two different agents.
You will not see the full discussion, only at most the last {judge_window} statements from each agent.
You will score how much the agents agree with one another on a scale from 1 to 5, where 5 means they agree completely and 1 means they disagree completely.
If they both give pros and cons regarding the question and neither takes a strong stance, then give them a 3.

Calibration examples:
Score 1: Agent1 strongly supports the topic and Agent2 strongly opposes it; both reject the other person's view.
Score 2: Agent1 and Agent2 mostly disagree, but one acknowledges a small point from the other.
Score 3: Agent1 and Agent2 are mixed, uncertain, or both present pros and cons without a clear shared stance.
Score 4: Agent1 and Agent2 mostly agree, with a minor caveat or difference in emphasis.
Score 5: Agent1 and Agent2 clearly agree and express the same stance on the topic.

Respond with an integer number only. Your response should contain no words, only a number.
If the sequence is empty, containing only the string ' ', then return -1."""


def judge_shim(judge_int, judge_window):
    if judge_int == 0:
        return judge_0(judge_window)
