class StagController extends Controller {
    constructor(stag) {
        super(stag);
        
    }

    // the stag either moves towards the nearest agent (default) or takes a random move.
    move(state) {
        var coin = Math.random();
        if (coin < 1/3) {
            var comuter_pos = [state[1], state[0]];
            var human_pos = [state[3], state[2]];
            var computer_distance = this.distance(comuter_pos);
            var human_distance = this.distance(human_pos);

            if(computer_distance < human_distance) {
                return this.takeActionTo(comuter_pos);
            }
            else {
                return this.takeActionTo(human_pos);
            }
        }
        else if (coin > 1/3 && coin < 1/3 + 0.254) {
            // 0.254 + 0.0825 ~ 0.33
            return 32; // stay
        }
        else {
            // 5 actions (0.2 for an action)
            // 1 - 1/3 - 0.254 = 619/1500
            // 619/1500 * 0.2 = 0.0825
            var action =  this.random();
            return action;
        }
    }
}