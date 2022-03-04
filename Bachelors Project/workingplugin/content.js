var ids = new Map();
var i = 0;
var L =  document.images.length;


for (let i = 0; i < L; i++){
    
    setTimeout(function() {
        // Add tasks to do
        var img_url = document.images.item(i).src;
        ids.set(decodeURIComponent(img_url), i.toString());
    $.ajax({
        url: "http://127.0.0.1:5000/" + img_url, 
        success: function(data) {
            if(data[0]['caption']!='error generating caption'){
                var numb = ids.get(data[0]['url']);
                var recv_url = document.images.item(numb);
                document.images.item(numb).alt = document.images.item(numb).alt.concat(". Generated caption: ",data[0]['caption']);

                swal.fire({title:'',
                    text:data[0]['caption'],
                    confirmButtonText: 'Click to Speak',
                    imageUrl:document.images.item(numb).src,
                    imageWidth: 400,
                    imageHeight: 200,
                    allowOutsideClick: false,
                
                }).then(function(){

                var msg = new SpeechSynthesisUtterance();
                    msg.text = data[0]['caption']; 
                    window.speechSynthesis.speak((msg));
                    swal.fire({title:'',
                    text:data[0]['caption'],
                    imageUrl:document.images.item(numb).src,
                    imageWidth: 400,
                    imageHeight: 200,
                    allowOutsideClick: false})
                });
                
            }  
        },
        error: function() {
        //   alert("There was an error contacting the servers to process your image");
        }
    });

    }, 7000 * i);
    
    
}






