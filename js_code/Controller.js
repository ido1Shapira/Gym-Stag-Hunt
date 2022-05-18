class Controller {
    constructor(controlled) {
        this.controlled = controlled;
        
    }

    validAction(action) {
        switch(action) {
            case 38: //up
                return this.controlled.tileFrom[1]>0 && this.controlled.tileFrom[1]-1>0;
            case 40: //down
                return this.controlled.tileFrom[1]<(mapH-1)&& this.controlled.tileFrom[1]+1<(mapH-1);
            case 37: //left
                return this.controlled.tileFrom[0]>0 && this.controlled.tileFrom[0]-1>0;
            case 39: //right
                return this.controlled.tileFrom[0]<(mapW-1) && this.controlled.tileFrom[0]+1 <(mapW-1);
            default:
                return false;
        }
    }
    getValidActions() {
        var actions = Object.keys(this.controlled.keysDown).map((i) => Number(i));
        var valid_actions = [];
        for(var i=0; i<actions.length; i++) {
            if(this.validAction(actions[i])) {
                valid_actions.push(actions[i]);
            }
        }
        return valid_actions;
    }

    // The distance between of the object controlled from pos
    distance(target_pos) {
        return Math.sqrt( Math.pow( (target_pos[0] - this.controlled.tileFrom[0]) , 2) + Math.pow( (target_pos[1] - this.controlled.tileFrom[1]) , 2) );
    }
    
    // take a action that make it close to the target
    takeActionTo(to) {
        if(this.controlled.tileFrom[1] < to[1]) {
            return 40 //down
        }
        else if(this.controlled.tileFrom[1] > to[1]){
            return 38 //up
        }
        else if(this.controlled.tileFrom[0] < to[0]) {
            return 39 //right
        }
        else if(this.controlled.tileFrom[0] > to[0]) {
            return 37 //left
        }
        return 32 //stay
    }

    random() {
        var valid_actions = this.getValidActions();
        var randomAction = valid_actions[Math.floor(valid_actions.length * Math.random())];
        return randomAction;
    }

    move(state) {
       return this.random();
    }
}