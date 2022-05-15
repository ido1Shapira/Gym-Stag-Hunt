class StagController {
    constructor(stag) {
        this.stag = stag;
        
    }

    validAction(action, board) {
        switch(action) {
            case 38: //up
                return this.stag.tileFrom[1]>0 && board[this.stag.tileFrom[1]-1][this.stag.tileFrom[0]]==1;
            case 40: //down
                return this.stag.tileFrom[1]<(mapH-1) && board[this.stag.tileFrom[1]+1][this.stag.tileFrom[0]]==1;
            case 37: //left
                return this.stag.tileFrom[0]>0 && board[this.stag.tileFrom[1]][this.stag.tileFrom[0]-1]==1;
            case 39: //right
                return this.stag.tileFrom[0]<(mapW-1) && board[this.stag.tileFrom[1]][this.stag.tileFrom[0]+1]==1;
            default:
                return false;
        }
    }
    getValidActions(board) {
        var actions = Object.keys(this.stag.keysDown).map((i) => Number(i));
        var valid_actions = [];
        for(var i=0; i<actions.length; i++) {
            if(this.validAction(actions[i], board)) {
                valid_actions.push(actions[i]);
            }
        }
        return valid_actions;
    }

    distance(pos) {
        return Math.sqrt( Math.pow( (pos[0] - this.stag.position[0]) , 2) + Math.pow( (pos[1] - this.stag.position[1]) , 2) );
    }

    whereis(sub_state) {
        var indexs = [];
        sub_state.filter(function(row, i){
            row.filter(function(row, j) {
                if(row == 1) {
                    indexs.push([j,i]);
                }
            });
        });
        return indexs;
    }
    takeActionTo(from, to) {
        // take a action that make it close to the award

        // maybe use here astar to find the shortest path:
        // https://github.com/bgrins/javascript-astar
        // or this:
        // https://github.com/rativardhan/Shortest-Path-2D-Matrix-With-Obstacles
        if(from[1] < to[1]) {
            return 40 //down
        }
        else if(from[1] > to[1]){
            return 38 //up
        }
        else if(from[0] < to[0]) {
            return 39 //right
        }
        else if(from[0] > to[0]) {
            return 37 //left
        }
        throw "takeActionTo("+from +","+ to+ "): could not found the action";
    }

    //////////////////////////////////////////////////////////////////////////////////
    random(state) {
        var valid_actions = this.getValidActions(state[0]);
        var randomAction = valid_actions[Math.floor(valid_actions.length * Math.random())];
        return randomAction;
    }

    closest(state) {
        var stag_pos = this.stag.tileFrom;
        var all_positions = [this.whereis(state[1]), this.whereis(state[2])]; // whereis other players
        
        const SPA = new ShortestPathAlgo(state[0]);
        
        SPA.run(stag_pos, all_positions[0]);
        var min_d = SPA.getMinDistance();
        var min_path = SPA.getSortestPath();
        
        for (var i=1; i<all_positions.length; i++) {
            var award_pos = all_positions[i];
            SPA.run(stag_pos, award_pos);
            var d = SPA.getMinDistance();
            if(d < min_d) {
                min_d = d;
                min_path = SPA.getSortestPath();
            }
        }
        return this.takeActionTo(min_path[0], min_path[1]);
    }


    // the stag either moves towards the nearest agent (default) or takes a random move.
    move(state) {
        var coin = Math.random();
        if (coin < 0.5) {
            return this.random(state);
        }
        else {
            return this.random(state);
        }
    }
}