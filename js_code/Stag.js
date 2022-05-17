class Stag {
    constructor(tileFrom, tileTo, dimensions, position, delayMove, iconPath, gameMap, tileW, tileH, mapW, mapH) {
        
        this.tileFrom	= tileFrom;
        this.tileTo		= tileTo;
        this.timeMoved	= 0;
        this.position	= position;
        this.delayMove	= delayMove;
        this.dimensions	= dimensions;
        
        this.iconImage = new Image();
        this.iconImage.src = ('data/assets/entities/'+iconPath);

        this.tileW = tileW;
        this.tileH = tileH;

        this.mapW = mapW;
        this.mapH = mapH;

        this.gameMap = gameMap;
    
    }

    keysDown = {
        32 : false, //stay
        37 : false, //left
        38 : false, //up
        39 : false, //right
        40 : false, //down
    };

    placeAt(x,y) {
        this.tileFrom	= [x,y];
	    this.tileTo		= [x,y];
	    this.position	= [((tileW*x)+((tileW-this.dimensions[0])/2)),
		    ((tileH*y)+((tileH-this.dimensions[1])/2))];
    }

    processMovement(t) {
        if(this.keysDown[32]) {
            if((t-this.timeMoved)<this.delayMove)
            {
                return true;
            }
        }
        if(this.tileFrom[0]==this.tileTo[0] && this.tileFrom[1]==this.tileTo[1]) { return false; }

        if((t-this.timeMoved)>=this.delayMove)
        {
            this.placeAt(this.tileTo[0], this.tileTo[1]);
        }
        else
        {
            this.position[0] = (this.tileFrom[0] * tileW) + ((tileW-this.dimensions[0])/2);
            this.position[1] = (this.tileFrom[1] * tileH) + ((tileH-this.dimensions[1])/2);

            if(this.tileTo[0] != this.tileFrom[0])
            {
                var diff = (tileW / this.delayMove) * (t-this.timeMoved);
                this.position[0]+= (this.tileTo[0]<this.tileFrom[0] ? 0 - diff : diff);
            }
            if(this.tileTo[1] != this.tileFrom[1])
            {
                var diff = (tileH / this.delayMove) * (t-this.timeMoved);
                this.position[1]+= (this.tileTo[1]<this.tileFrom[1] ? 0 - diff : diff);
            }

            this.position[0] = Math.round(this.position[0]);
            this.position[1] = Math.round(this.position[1]);           
        }

        return true;

    }

    place_position() {
        return [Math.round((this.tileFrom[0] * this.tileW) + ((this.tileW-this.dimensions[0])/2)),
                Math.round((this.tileFrom[1] * this.tileH) + ((this.tileH-this.dimensions[1])/2))];
    }

    set_in_random_tile(forbidden_pos) {
        var mapW = this.mapW;
        var mapH = this.mapH;
        var indexs = [];
        gameMap.filter(function(elem, index, array){
            if(elem == 1) {
                var i = Math.floor(index / mapW);
                var j = index % mapH;
                var forbidden_pos_flag = true
                for(var pos of forbidden_pos) {
                    forbidden_pos_flag &= (j != pos[0] && i != pos[1])
                }
                if(forbidden_pos_flag) {
                        // award can not be in player tile
                    indexs.push([j,i]);
                }
            }
        });
        // choose randomly 1 indexs
        var idx = Math.floor(Math.random() * indexs.length);
        this.tileFrom[0] = indexs[idx][0];
        this.tileFrom[1] = indexs[idx][1];

        this.tileTo[0] = indexs[idx][0];
        this.tileTo[1] = indexs[idx][1];

        this.position = this.place_position();
    }
}