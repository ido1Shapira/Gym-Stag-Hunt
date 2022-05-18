class StagController extends Controller {
    constructor(stag) {
        super(stag);
        
    }

    // the stag either moves towards the nearest agent (default) or takes a random move.
    move(state) {
        var coin = Math.random();
        if (coin < 0.5) {
            return this.random();
        }
        else {
            var comuter_pos = [state[0], state[1]];
            var human_pos = [state[2], state[3]];
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