//
// Author: Ido Shapira
// date: 15/05/2022
//
var ctx = null;
var canvas = null;
var gameMap = [
	0, 0, 0, 0, 0, 0, 0,
	0, 1, 1, 1, 1, 1, 0,
	0, 1, 1, 1, 1, 1, 0,
	0, 1, 1, 1, 1, 1, 0,
	0, 1, 1, 1, 1, 1, 0,
	0, 1, 1, 1, 1, 1, 0,
	0, 0, 0, 0, 0, 0, 0,
];
var tileW = 70, tileH = 70;
var mapW = 7, mapH = 7;
var dimensions_objects = [60,60]

var finishFlag = false;
var steps_per_game = 60; // How many episodes (time steps) occur during a single game of Stag Hunt before entity positions are reset and the game is considered done.

// reward function:
const stag_reward = 4; // Reinforcement reward for when agents catch the stag by occupying the same cell as it at the same time. Expected to be positive.
const forage_reward = 1; //Reinforcement reward for harvesting a plant. Expected to be positive.
const mauling_punishment = 0; //Reinforcement reward (or, rather, punishment) for getting mauled by a stag. Expected to be negative.

// time:
var currentSecond = 0, frameCount = 0, framesLastSecond = 0, lastFrameTime = 0;

// initialize:

function place_position(tile, dimensions) {
	return [Math.round((tile[0] * tileW) + ((tileW-dimensions[0])/2)),
			Math.round((tile[1] * tileH) + ((tileH-dimensions[1])/2))];
}

                    		// tileFrom, dimensions,       position,                              delayMove, icon
var human_player = new Player([1,1], [1,1], dimensions_objects, place_position([1,1], dimensions_objects), 500, "red_agent.png");
initializeFirebase();
var computer_player = new Player([5,1], [5,1], dimensions_objects, place_position([5,1], dimensions_objects), 500, "blue_agent.png");
var computer_controller = null;

firebase.database().ref("chosen-controller").once('value',
(snap) => {
	selectedBehavior = snap.val();
	computer_controller = new PlayerController(computer_player, selectedBehavior);
	var type = computer_controller.getType();
	// Generate a reference to a new location and add some data using push()
	var newPostRef = firebase.database().ref("all-games").push({
		behavior: type,
		realScore: true
	});
	// Get the unique ID generated by push() by accessing its key
	postID = newPostRef.key;
	// console.log("postID: "+postID);
});


// init stag
var stag = new Stag([3,3], [3,3], dimensions_objects, place_position([3,3], dimensions_objects), 500, "stag.png", gameMap, tileW, tileH, mapW, mapH);
stag_controller = new StagController(stag, "random");

// init shrubs
var shrubs = [];
for(var i=0; i<3; i++) {
	var new_shrub = new Shrub(dimensions_objects, "plant_fruit.png", gameMap, tileW, tileH, mapW, mapH);
	new_shrub.set_in_random_tile([computer_player.tileFrom, human_player.tileFrom, stag.tileFrom], shrubs);
	shrubs.push(new_shrub);
}


function toIndex(x, y)
{
	return((y * mapW) + x);
}

function zeros(dimensions) { // dimensions = [r,c] 
    var array = [];
    for (var i = 0; i < dimensions[0]; ++i) {
        array.push(dimensions.length == 1 ? 0 : zeros(dimensions.slice(1)));
    }
    return array;
}

function getState() {
// Coords gets you a coordinate array with boolean tuples of size 4 signifying the presence of entities in that cell
// (index 0 is agent A, index 1 is agent B, index 2 is stag, index 3 is plant).

	var state = [
				[computer_player.tileFrom[1], computer_player.tileFrom[0]],
				[human_player.tileFrom[1], human_player.tileFrom[0]],
				[stag.tileFrom[1], stag.tileFrom[0]]
			];

	for(var shrub of shrubs) {
		state.push([shrub.tile[1], shrub.tile[0]]);
	}
	return state.flat();
}

var moved = false;
var humanMove = null;
var computerMove = null;
var stagMove = null;

var handleKeyUp = function(e) {
	if(e.keyCode>=37 && e.keyCode<=40) {
		var currentFrameTime = Date.now();
		if((currentFrameTime-human_player.timeMoved>=human_player.delayMove)) {
			var validHumanAction = false;
			switch(e.keyCode) {
				case 37:
					if(human_player.tileFrom[0]>0 && gameMap[toIndex(human_player.tileFrom[0]-1, human_player.tileFrom[1])]==1) {
						human_player.tileTo[0]-= 1; //left
						validHumanAction = true;
						humanMove = e.keyCode;
					}
					break;
				case 38:
					if(human_player.tileFrom[1]>0 && gameMap[toIndex(human_player.tileFrom[0], human_player.tileFrom[1]-1)]==1) {
						human_player.tileTo[1]-= 1; //up
						validHumanAction = true;
						humanMove = e.keyCode;
					}
					break;
				case 39:
					if(human_player.tileFrom[0]<(mapW-1) && gameMap[toIndex(human_player.tileFrom[0]+1, human_player.tileFrom[1])]==1) {
						human_player.tileTo[0]+= 1; //right
						validHumanAction = true;
						humanMove = e.keyCode;
					}
					break;
				case 40:
					if(human_player.tileFrom[1]<(mapH-1) && gameMap[toIndex(human_player.tileFrom[0], human_player.tileFrom[1]+1)]==1) {
						human_player.tileTo[1]+= 1; //down
						validHumanAction = true;
						humanMove = e.keyCode;
					}
					break;
			}
			human_player.timeMoved = currentFrameTime;
			if (humanMove != null) {
				human_player.keysDown[humanMove] = true;
			}

			if(validHumanAction) {
				//blue player move
				var state = getState();
				computerMove = computer_controller.move(state);
				computer_player.keysDown[computerMove] = true;

				// the stag either moves towards the nearest agent (default) or takes a random move.
				stagMove = stag_controller.move(state);
				stag.keysDown[stagMove] = true;

				saveToFirebase(state, humanMove, computerMove, stagMove);
				steps_per_game--;
			}
		}
	}
}

window.onload = function()
{
	canvas = document.getElementById('game');
	ctx = canvas.getContext("2d");
	requestAnimationFrame(drawGame);
	ctx.font = "bold 10pt sans-serif";

	// window.addEventListener("keyup", handleKeyUp);
};

function logics()
{
	var currentFrameTime = Date.now();
	var timeElapsed = currentFrameTime - lastFrameTime;
	
	var sec = Math.floor(Date.now()/1000);
	if(sec!=currentSecond)
	{
		currentSecond = sec;
		framesLastSecond = frameCount;
		frameCount = 1;
	}
	else { frameCount++; }

	//human move
	if(!human_player.processMovement(currentFrameTime)) //move human player on board
	{
		moved = true;
		human_player.keysDown[humanMove] = false;

		if(human_player.tileFrom[0] != human_player.tileTo[0] || human_player.tileFrom[1] != human_player.tileTo[1])
		{ human_player.timeMoved = currentFrameTime;}
	}
	
	//computer mvoe
	if(!computer_player.processMovement(currentFrameTime)) //move computer player on board
	{
		if(computer_player.keysDown[38] && computer_player.tileFrom[1]>0 && gameMap[toIndex(computer_player.tileFrom[0], computer_player.tileFrom[1]-1)]==1) { computer_player.tileTo[1]-= 1; }
		else if(computer_player.keysDown[40] && computer_player.tileFrom[1]<(mapH-1) && gameMap[toIndex(computer_player.tileFrom[0], computer_player.tileFrom[1]+1)]==1) { computer_player.tileTo[1]+= 1; }
		else if(computer_player.keysDown[37] && computer_player.tileFrom[0]>0 && gameMap[toIndex(computer_player.tileFrom[0]-1, computer_player.tileFrom[1])]==1) { computer_player.tileTo[0]-= 1; }
		else if(computer_player.keysDown[39] && computer_player.tileFrom[0]<(mapW-1) && gameMap[toIndex(computer_player.tileFrom[0]+1, computer_player.tileFrom[1])]==1) { computer_player.tileTo[0]+= 1; }
		computer_player.keysDown[computerMove] = false;

		if(computer_player.tileFrom[0]!=computer_player.tileTo[0] || computer_player.tileFrom[1]!=computer_player.tileTo[1])
		{ computer_player.timeMoved = currentFrameTime; }
	}

	//stag move
	if(!stag.processMovement(currentFrameTime)) //move stag on board
	{
		if(stag.keysDown[38] && stag.tileFrom[1]>0 && gameMap[toIndex(stag.tileFrom[0], stag.tileFrom[1]-1)]==1) { stag.tileTo[1]-= 1; }
		else if(stag.keysDown[40] && stag.tileFrom[1]<(mapH-1) && gameMap[toIndex(stag.tileFrom[0], stag.tileFrom[1]+1)]==1) { stag.tileTo[1]+= 1; }
		else if(stag.keysDown[37] && stag.tileFrom[0]>0 && gameMap[toIndex(stag.tileFrom[0]-1, stag.tileFrom[1])]==1) { stag.tileTo[0]-= 1; }
		else if(stag.keysDown[39] && stag.tileFrom[0]<(mapW-1) && gameMap[toIndex(stag.tileFrom[0]+1, stag.tileFrom[1])]==1) { stag.tileTo[0]+= 1; }
		else if(human_player.keysDown[32]) { }
		stag.keysDown[stagMove] = false;

		if(stag.tileFrom[0]!=stag.tileTo[0] || stag.tileFrom[1]!=stag.tileTo[1])
		{
			stag.timeMoved = currentFrameTime;
		}
	}

	//check for overlaps shrubs
	for(var i=0; i<shrubs.length; i++) {
		var overlaps = false; 
		var temp_shrub = shrubs[i];
		if(temp_shrub.tile[0] == human_player.tileTo[0] && temp_shrub.tile[1] == human_player.tileTo[1]
			& (currentFrameTime-human_player.timeMoved)>=human_player.delayMove) {
			overlaps = true;
			human_player.score = human_player.score + forage_reward;
			// if(temp_shrub.tile[0] == computer_player.tileFrom[0] && temp_shrub.tile[1] == computer_player.tileFrom[1]) {
			// 	computer_player.score = computer_player.score + forage_reward;
			// }
		}
		if(temp_shrub.tile[0] == computer_player.tileTo[0] && temp_shrub.tile[1] == computer_player.tileTo[1]
			& (currentFrameTime-human_player.timeMoved)>=human_player.delayMove) {
			overlaps = true;
			computer_player.score = computer_player.score + forage_reward;
		}

		if(overlaps) {
			shrubs[i].set_in_random_tile([computer_player.tileFrom, human_player.tileFrom, stag.tileFrom], shrubs);
		}
	}

	//check for overlaps stag
	if(stag.tileTo[0] == human_player.tileTo[0] && stag.tileTo[1] == human_player.tileTo[1]) {
		if(stag.tileTo[0] == computer_player.tileTo[0] && stag.tileTo[1] == computer_player.tileTo[1]
			& (currentFrameTime-human_player.timeMoved)>=human_player.delayMove) {
			human_player.score = human_player.score + stag_reward;
			computer_player.score = computer_player.score + stag_reward;
			stag.set_in_random_tile([computer_player.tileFrom, human_player.tileTo, stag.tileTo], shrubs);
		}
		// else {
		// 	human_player.score = human_player.score + mauling_punishment;		
		// }
	}
	// if(stag.tileFrom[0] == computer_player.tileFrom[0] && stag.tileFrom[1] == computer_player.tileFrom[1]) {
	// 	if(stag.tileFrom[0] != human_player.tileFrom[0] || stag.tileFrom[1] != human_player.tileFrom[1]) {
	// 		computer_player.score = computer_player.score + mauling_punishment;
	// 	}
	// }

	// check if the game finished
	if(steps_per_game == 0 && !finishFlag && (currentFrameTime-human_player.timeMoved)>=human_player.delayMove) {
				
		console.log('1) Computer score: '+ computer_player.score);
		console.log('2) Human score: '+ human_player.score);

		finishFlag = true;
		window.removeEventListener("keyup", handleKeyUp);
		finishGame();
	}
	
	lastFrameTime = currentFrameTime;
}

function drawGame()
{
	if(ctx==null) { return; }
	logics();

	// draw objects:
	for(var y = 0; y < mapH; ++y) // draw the board
	{
		for(var x = 0; x < mapW; ++x)
		{
			switch(gameMap[((y*mapW)+x)])
			{
				case 0:
					ctx.fillStyle = "#685b48"; // color: brown
					break;
				default:
					ctx.fillStyle = "#C4A484"; // color: light brown
			}
			ctx.strokeRect(x*tileW, y*tileH, tileW, tileH);
			ctx.fillRect(x*tileW, y*tileH, tileW, tileH);
		}
	}
	
	var borderWidth = 1;   
	var offset = borderWidth * 2;
	
	for(shrub of shrubs) {
		ctx.drawImage(shrub.iconImage, shrub.position[0] - borderWidth, shrub.position[1] -borderWidth, shrub.dimensions[0] + offset, shrub.dimensions[1] + offset);
	}
	ctx.drawImage(computer_player.iconImage, computer_player.position[0] - borderWidth, computer_player.position[1] -borderWidth, computer_player.dimensions[0] + offset, computer_player.dimensions[1] + offset);
	ctx.drawImage(human_player.iconImage, human_player.position[0] - borderWidth, human_player.position[1] -borderWidth, human_player.dimensions[0] + offset, human_player.dimensions[1] + offset);

	ctx.drawImage(stag.iconImage, stag.position[0] - borderWidth, stag.position[1] -borderWidth, stag.dimensions[0] + offset, stag.dimensions[1] + offset);

	ctx.fillStyle = "#0000ff"; // title color: blue
	ctx.fillText("Blue score: " + computer_player.score, 360, 25);
    ctx.fillText("Blue action: " + computer_player.codeToAction[computerMove], 360, 50);

	ctx.fillStyle = "#ff0000"; // title color: red
	ctx.fillText("Red score: " + human_player.score, 20, 25);
	ctx.fillText("Red action: " + human_player.codeToAction[humanMove], 20, 50);

	ctx.fillStyle = "#ffffff"; // title color: red
	ctx.fillText("Steps left: " + steps_per_game, 20, 450);
	
	requestAnimationFrame(drawGame);
}
