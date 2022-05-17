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
            case 32: //stay
                return true;
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
        return Math.sqrt( Math.pow( (pos[0] - this.stag.tileFrom[0]) , 2) + Math.pow( (pos[1] - this.stag.tileFrom[1]) , 2) );
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
    
    // take a action that make it close to the target
    takeActionTo(from, to) {
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
        return 32 //stay
    }

    // the stag either moves towards the nearest agent (default) or takes a random move.
    move(state) {
        var coin = Math.random();
        if (coin < 0.5) {
            var valid_actions = this.getValidActions(state[0]);
            var randomAction = valid_actions[Math.floor(valid_actions.length * Math.random())];
            return randomAction;
        }
        else {
            var comuter_pos = this.whereis(state[1])[0];
            var human_pos = this.whereis(state[2])[0];
            var computer_distance = this.distance(comuter_pos);
            var human_distance = this.distance(human_pos);

            if(computer_distance < human_distance) {
                return this.takeActionTo(this.stag.tileFrom, comuter_pos);
            }
            else {
                return this.takeActionTo(this.stag.tileFrom, human_pos);
            }
        }
    }
}