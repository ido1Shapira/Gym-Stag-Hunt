class StagController extends Controller {
    constructor(stag) {
        super(stag);
        
    }

    // the stag either moves towards the nearest agent (default) or takes a random move.
    move(state) {
        var coin = Math.random();
        if (coin < 0.5) {
            var action =  this.random();
            return action;
        }
        else {
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
    }
}