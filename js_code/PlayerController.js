class PlayerController extends Controller{
    TYPES = {
        "random": false,
        "follow_stag": false,
        "closest": false,

        // "ddqn": false,
        // "sarl ddqn": false,
    }

    toAction = {
        1: 37, //left
        2: 38, //up
        3: 39, //right
        4: 40, //down
    }

    constructor(player, type) {
        // 3 kinds of type:
        // 1. random - Moves randomly
        // 2. closest - Go to the closest shrub
        // 3. follow_stag - Follow the stag at each move
        super(player)

        if(type == -1) {
            var all = Object.keys(this.TYPES);
            type = all[Math.floor(all.length * Math.random())];
        }
        if(type == -2) {
            var baselines = Object.keys(this.TYPES).slice(0,3);
            type = baselines[Math.floor(baselines.length * Math.random())];
        }
        // if(type == -3) {
        //     var ddqns = Object.keys(this.TYPES).slice(3,5);
        //     type = ddqns[Math.floor(ddqns.length * Math.random())];
        // }
    
        this.TYPES[type] = true;
        this.type = type;

        this.loadAgent();
    }
    
    getType() { return this.type; }

    move(state) {
        switch(this.type) {
            case "random":
                return this.random();
            case "follow_stag":
                return this.follow_stag(state);
            case "closest":
                return this.closest(state);

            // case "ddqn": case "sarl ddqn":
            //     return this.predict(state);
            default:
                throw "move(state): not a valid baseline"
        }
    }

    ////////////////////////////// All baselines ////////////////////////////////////////////////
    
    closest(state) {
        var shrubs_pos = [[state[7], state[6]], [state[9], state[8]], [state[11], state[10]]];
        var idx = 0;
        var min_dis = this.distance(shrubs_pos[0]);
        for(var i=1; i<shrubs_pos.length; i++) {
            var temp = this.distance(shrubs_pos[i]);
            if(min_dis > temp) {
                min_dis = temp;
                idx = i;
            }
        }
        return this.takeActionTo(shrubs_pos[idx]);
    }

    follow_stag(state) {
        var stag_pos = [state[5], state[4]];
        return this.takeActionTo(stag_pos);
    }
    
    ////////////////////////////// Advance agents ////////////////////////////////////////////////
    loadAgent() {
        var deepRL = true;
        var path = 'data/models/';
        switch(this.type) {
            case "ddqn":
                path += 'ddqn_agent';
                break;
            case "sarl ddqn":
                path += 'SARL_ddqn_agent_0.4';
                break;
            
            default:
                deepRL = false;

        }
        if(deepRL) {
            (async () => {
                // this.model = await tf.loadLayersModel(path + '/model.json');
                this.model = await tf.loadGraphModel(path + '/model.json');
            })()
        }
    }

    // functions for numjs that were missing
    divideByScalar(arr, scalar) {
        var nj_matrix = nj.array(arr, 'float32');
        if(scalar < 1) {
            throw "scalar: " + scalar + " can not be negitive";
        }
        var scalar_matrix = nj.ones(nj_matrix.shape, 'float32').multiply(scalar);
        return nj.divide(nj_matrix, scalar_matrix);;
    }
    argmax(dict) {
        //return key of max value
        var max_key=0;
        var max_value=dict[max_key];
        for(var key in dict) {
            if(dict[key] > max_value) {
                max_value = dict[key];
                max_key = key;
            }
        }
        return max_key;
    }
    slice(arr, from, to) {
        var result = [];
        for(var j=from; j<to; j++) {
            result.push(arr[j]);
        }
        return result;
    }

    preproccess(state) {
        var r = nj.add(
            this.divideByScalar(state[this.toIndex['Human awards']], 2),
            nj.add(
                state[this.toIndex['Human trace']],
                state[this.toIndex['All awards']]));
        
        var g = nj.add(
            this.divideByScalar(state[this.toIndex['Board']] , 3),
            nj.array(state[this.toIndex['All awards']], 'float32'));
        
        var b = nj.add(
            this.divideByScalar(state[this.toIndex['Computer awards']], 2),
            nj.add(
                nj.array(state[this.toIndex['Computer trace']], 'float32'),
                nj.array(state[this.toIndex['All awards']], 'float32')));

        var rgb = nj.stack([b, g, r], -1, 'float32');
        
        // NormalizeImage
        var min_matrix = nj.ones(rgb.shape, "float32").multiply(nj.min(rgb));
        var max_matrix = nj.ones(rgb.shape, "float32").multiply(nj.max(rgb));
        rgb = nj.divide(nj.subtract(rgb, min_matrix), nj.subtract(max_matrix, min_matrix)).tolist();
        
        state = nj.stack([state[0], state[1], state[2],
                       state[3],state[4],state[5]], -1, 'float32').tolist();
        
        return rgb;
    }
    predict(state) {
        var img = this.preproccess(state);
        var tensorImg = tf.tensor3d(img).expandDims(0);
        var score = this.model.predict(tensorImg).dataSync();
        var dict_scores = {
            0: score[0], //stay
            1: score[1], //left
            2: score[2], //up
            3: score[3], //right
            4: score[4], //down
        }
        var action = this.argmax(dict_scores);
        while(! this.validAction(this.toAction[action], state[0])) {
            delete dict_scores[action];
            action = this.argmax(dict_scores);
        }

        
        return this.toAction[action];
    }  
}