var remote = require('electron').remote,
arguments = remote.getGlobal('sharedObject').prop1;
document.getElementById("user_name").innerHTML="Daniel"
