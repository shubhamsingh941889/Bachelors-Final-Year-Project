var callback = function (results) {
    // ToDo: Do something with the image urls (found in results[0])

    var ultimateHttp="";
    for(var i=0;i<4;i++){
      ultimateHttp=ultimateHttp.concat(results[0][i][0]);
      ultimateHttp=ultimateHttp.concat(",");
    }
    var ultimateCaption="";
    alert(ultimateHttp);
    //alert(results[0][1][0]);
    $.ajax({
      url: "https://v0fkjw6l82.execute-api.us-west-2.amazonaws.com/prod/auto-alt-text-api?url=" + ultimateHttp,
      //url: "https://v0fkjw6l82.execute-api.us-west-2.amazonaws.com/prod/auto-alt-text-api?url=" + https://en.wikipedia.org/wiki/File:Angels_Stadium.JPG,
      success: function(data) {
        for(var i=0;i<4;i++){
        var caption = data[i].captions[0];
        if(caption.prob != -1) {
          ultimateCaption=ultimateCaption.concat(data[i].captions[0].sentence+",");
          //alert(data[1].captions[0].sentence+"hi");
        } else {
          ultimateCaption=ultimateCaption.concat("error,");
          //alert("probability is less");
        }
      }
      alert(ultimateCaption);
      var getAltTags=ultimateCaption.split(",");
      alert("hi");
      alert(getAltTags[1]);
      alert("hi");
      document.body.innerHTML = '';
      for (var i=0;i<4;i++) {
          var img = document.createElement('img');
          img.src = results[0][i][0];
          //img.alt = resutls[0][i][1];
          img.alt = getAltTags[i];
          //alert(img.src);
        alert(img.src+" hi "+img.alt);
          document.body.appendChild(img);
      }
      },
      error: function() {
        alert("There was an error contacting the servers to process your image");
      }
    });
alert("hi");


};

chrome.tabs.onUpdated.addListener( function (tabId, changeInfo, tab) {
  if (changeInfo.status == 'complete' && tab.active) {

    chrome.tabs.query({ // Get active tab
        active: true,
        currentWindow: true
    }, function (tabs) {
        chrome.tabs.executeScript(tabs[0].id, {
            code: 'Array.prototype.map.call(document.images, function (i) {var getImage=[i.src,i.alt]; return getImage;});'
        }, callback);
    });

  }
})
