class Player{
    codeToAction = {
        null : "",
        37 : "left",
        38 : "up",
        39 : "right",
        40 : "down",
    };
    keysDown = {
        37 : false, //left
        38 : false, //up
        39 : false, //right
        40 : false, //down
    };
    scores = {
        start : 0.0,
        // stay : -0.01, //-1,
        // step : -0.05, //-5,
        // finish : 1.0 //100,
    }
    
    constructor(tileFrom , tileTo, timeMoved, dimensions, position, delayMove, iconPath) {
        this.tileFrom	= tileFrom;
        this.tileTo		= tileTo;
        this.timeMoved	= timeMoved;
        this.dimensions	= dimensions;
        this.position	= position;
        this.delayMove	= delayMove;
        this.score = this.scores.start;
        this.iconImage = new Image();
        this.iconImage.src = ('data/assets/entities/'+iconPath);
    }
    placeAt(x,y) {
        this.tileFrom	= [x,y];
	    this.tileTo		= [x,y];
	    this.position	= [((tileW*x)+((tileW-this.dimensions[0])/2)),
		    ((tileH*y)+((tileH-this.dimensions[1])/2))];
    }
    processMovement(t) {
        if(this.tileFrom[0]==this.tileTo[0] && this.tileFrom[1]==this.tileTo[1]) { return false; }


        if((t-this.timeMoved)>=this.delayMove)
        {
            this.placeAt(this.tileTo[0], this.tileTo[1]);

            // this.score = this.score + this.scores.step;
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
}