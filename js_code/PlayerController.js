class PlayerController extends Controller{
    TYPES = {
        "random": false,
        // "selfish": false,
        // "farthest": false,
        // "closest": false,
        // "TSP": false,
        // // "mix": false

        // "ddqn": false,
        // "sarl ddqn": false,

        // "ppo": false,
        // "sarl ppo": false,

        // "ddqn distribution": false,
        // "sarl ddqn distribution": false
    }

    toAction = {
        1: 37, //left
        2: 38, //up
        3: 39, //right
        4: 40, //down
    }

    constructor(player, type) {
        //5 kinds of type:
        // 1. random - Moves randomly
        // 2. selfish - stay in place
        // 3. farthest - Moves to the farthest award that its still closer than the other agent
        // 4. closest - Go to the closest award
        // 5. TSP - Moves by the solution of the TSP problem
        // 6. mix of all controllers
        
        // if(type == -1) {
        //     var all = Object.keys(this.TYPES);
        //     type = all[Math.floor(all.length * Math.random())];
        // }
        // if(type == -2) {
        //     var baselines = Object.keys(this.TYPES).slice(0,5);
        //     type = baselines[Math.floor(baselines.length * Math.random())];
        // }
        // if(type == -3) {
        //     var ddqns = Object.keys(this.TYPES).slice(5,7);
        //     type = ddqns[Math.floor(ddqns.length * Math.random())];
        // }
        // if(type == -4) {
        //     var ppos = Object.keys(this.TYPES).slice(7, 9);
        //     type = ppos[Math.floor(ppos.length * Math.random())];
        // }
        // if(type == -5) {
        //     var distribution = Object.keys(this.TYPES).slice(9,11);
        //     type = distribution[Math.floor(distribution.length * Math.random())];
        // }
        // if(type == -6) {
        //     var special = ['closest', 'random', 'farthest'];
        //     type = special[Math.floor(special.length * Math.random())];
        // }

        super(player)
        this.TYPES[type] = true;
        this.type = type;

        // this.loadAgent();
    }
    
    getType() { return this.type; }

    move(state) {
        switch(this.type) {
            case "random":
                return this.random();
            // case "selfish":
            //     return this.selfish(state);
            // case "closest":
            //     return this.closest(state);
            // case "farthest":
            //     return this.farthest(state);
            // case "TSP":
            //     return this.TSP(state);
            // case "mix":
            //     return this.mix(state);
            // case "ddqn": case "sarl ddqn":
            // case "ppo": case "sarl ppo":
            // case "ddqn distribution": case "sarl ddqn distribution":
            //     return this.predict(state);
            default:
                throw "move(state): not a valid baseline"
        }
    }

    ////////////////////////////// All baselines ////////////////////////////////////////////////
    
    // selfish(state) {
    //     return 32;
    // }
    closest(state) {
        var player_pos = this.player_controlled.tileFrom;
        var all_awards_positions = this.whereis(state[5]); // whereis all awards
        
        const SPA = new ShortestPathAlgo(state[0]);
        
        SPA.run(player_pos, all_awards_positions[0]);
        var min_d = SPA.getMinDistance();
        var min_path = SPA.getSortestPath();
        
        for (var i=1; i<all_awards_positions.length; i++) {
            var award_pos = all_awards_positions[i];
            SPA.run(player_pos, award_pos);
            var d = SPA.getMinDistance();
            // what happen when min_d == d ? maybe to consider the human player here
            // TODO: ask Amos
            if(d < min_d) {
                min_d = d;
                min_path = SPA.getSortestPath();
            }
        }
        return this.takeActionTo(min_path[0], min_path[1]);
    }
    farthest(state) {
        var player_pos = this.player_controlled.tileFrom;
        var all_awards_positions = this.whereis(state[5]); // whereis all awards
        var human_pos = this.whereis(state[1])[0];
        
        const SPA = new ShortestPathAlgo(state[0]);
        var max_d = Number.NEGATIVE_INFINITY;
        var max_path = null;

        for (var i=0; i<all_awards_positions.length; i++) {
            var award_pos = all_awards_positions[i];
            SPA.run(human_pos, award_pos);
            var d_h = SPA.getMinDistance(); //human distance
            SPA.run(player_pos, award_pos);
            var d_c = SPA.getMinDistance(); //computer distance
            if(d_c <= d_h) {
                //Check if the computer comes before the human
                if(d_c > max_d) {
                    max_d = d_c;
                    max_path = SPA.getSortestPath();
                }
            }
        }
        return this.takeActionTo(max_path[0], max_path[1]);
    }
    TSP(state) {
        var player_pos = this.player_controlled.tileFrom;
        var all_awards_positions = this.whereis(state[5]);
        all_awards_positions.unshift(player_pos);
        var map_cost = new Map();
        for(var point of all_awards_positions) {
            map_cost.set(point, new Map());
        }
        const SPA = new ShortestPathAlgo(state[0]);
        var permutations = generateCityRoutes(all_awards_positions);
        var min_cost = Number.POSITIVE_INFINITY;
        var awards_order = [];
        for(var path of permutations) {
            // console.log(p);
            var temp_cost = 0;
            for(var i=0; i<path.length-1; i++) {
                var point1 = path[i];
                var point2 = path[i+1];
                
                var cost = map_cost.get(point1).get(point2);
                if(cost === undefined) {
                    //insert cost
                    SPA.run(point1, point2);
                    var d = SPA.getMinDistance(); //computer distance
                    map_cost.get(point1).set(point2, d);
                    cost = map_cost.get(point1).get(point2);
                }
                temp_cost += cost;
            }
            if(temp_cost < min_cost) {
                min_cost = temp_cost;
                awards_order = path;
            }
        }
        SPA.run(awards_order[0], awards_order[1]);
        var optimal_path = SPA.getSortestPath();
        return this.takeActionTo(optimal_path[0], optimal_path[1]);
    }
    // mix(state) {
    //     var baselines = Object.keys(this.TYPES);
    //     const index = baselines.indexOf("mix");
    //     if (index > -1) {
    //         baselines.splice(index, 1);
    //     }
    //     var random_baseline = baselines[Math.floor(baselines.length * Math.random())];
    //     switch(random_baseline) {
    //         case "random":
    //             return this.random(state);
    //         case "selfish":
    //             return this.selfish(state);
    //         case "closest":
    //             return this.closest(state);
    //         case "farthest":
    //             return this.farthest(state);
    //         case "TSP":
    //             return this.TSP(state);
    //     }
    // }
    
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
            
            case "ddqn distribution":
                path += 'ddqn_agent_distribution';
                break;

            case "sarl ddqn distribution":
                path += 'SARL_ddqn_agent_0.615_distribution';
                break;

            case "ppo":
                // path += "ppo_actor_agent";
                // break;
                throw "file not found! at sarl ppo";
            case "sarl ppo":
                throw "file not found! at sarl ppo";
            
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