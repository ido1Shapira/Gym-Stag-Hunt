class Shrub {
    constructor(dimensions, iconPath, gameMap, tileW, tileH, mapW, mapH) {
        
        this.dimensions	= dimensions;
        
        this.iconImage = new Image();
        this.iconImage.src = ('data/assets/entities/'+iconPath);

        this.tileW = tileW;
        this.tileH = tileH;

        this.mapW = mapW;
        this.mapH = mapH;

        this.gameMap = gameMap;
        
        this.tile = [0, 0];
        this.position = [0, 0];
    }

    place_position() {
        return [Math.round((this.tile[0] * this.tileW) + ((this.tileW-this.dimensions[0])/2)),
                Math.round((this.tile[1] * this.tileH) + ((this.tileH-this.dimensions[1])/2))];
    }

    set_in_random_tile(forbidden_pos, shrubs) {
        for(var shrub of shrubs) {
            forbidden_pos.push(shrub.tile);
        }
        var mapW = this.mapW;
        var mapH = this.mapH;
        var indexs = [];
        gameMap.filter(function(elem, index, array){
            if(elem == 1) {
                var i = Math.floor(index / mapW);
                var j = index % mapH;
                var forbidden_pos_flag = true
                for(var pos of forbidden_pos) {
                    forbidden_pos_flag &= (j != pos[0] || i != pos[1]);
                }
                if(forbidden_pos_flag) {
                        // award can not be in player tile
                    indexs.push([j,i]);
                }
            }
        });
        // choose randomly 1 indexs
        var idx = Math.floor(Math.random() * indexs.length);
        this.tile[0] = indexs[idx][0];
        this.tile[1] = indexs[idx][1];

        this.position = this.place_position();
    }
}