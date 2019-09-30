function printIntentResponse(result) {
  $( "#fuzzy-intent-results" ).fadeOut(function() {
    $( "#fuzzy-intent-results" ).html(result);
  });
  $( "#fuzzy-intent-results" ).fadeIn();
}

$('#parse-fuzzy-intent').click(function(e){
  e.preventDefault();
  var user_text = $('#input-user-text').val();
  console.log(user_text);

  if (user_text != "") {
    $.ajax({
        type: 'POST',
        url: '/intentgateway/v2/intent-gateway',
        // data: JSON.stringify ({text: user_text, intent_context: ["claims", "claims_file_subreasons", "common_chat", "slots"]}),
        data: JSON.stringify ({text: user_text, version: "ig_phase_2"}),
        success: function(resp) { console.log(resp); 
          var rows = '';
          var entity_rows = '';
          var csv1 = []
          var csv2 = []
          $.each(resp['intents'], function(intent, data){

            $.each(data, function(intent_idx, matching_models){
              rows += "\
                <tr>\
                <td>" + intent+ "</td>\
                <td>" + JSON.stringify(matching_models) + "</td>\
                </tr>\
              "
              });
          });

          $.each(resp['datetime_entities'], function(entity, data){

            $.each(data, function(entity_idx, entity_info){
              entity_rows += "\
                <tr>\
                <td>" + entity_idx+ "</td>\
                <td>" + entity_info + "</td>\
                </tr>\
              "
            });
          });

        var result = "\
          <p>parsed user text: <strong>" + user_text + "</strong> as</p>\
          <table class='table table-bordered'>\
            <thead>\
              <tr>\
                <th>Intent</th>\
                <th>Intent Model</th>\
              </tr>\
            </thead>\
            <tbody>\
              " + rows + "\
            </tbody>\
          </table>\
          <p>\
          <table class='table table-bordered'>\
            <thead>\
              <tr>\
                <th>Match</th>\
                <th>Datetime Entity</th>\
              </tr>\
            </thead>\
            <tbody>\
              " + entity_rows + "\
            </tbody>\
          </table>\
          <p>Incorrect? your feedback allows us to improve AVA <a href='/intentgateway/demo-feedback/"+user_text+"'>please file an issue.</a></p>";
        printIntentResponse( result );

        },
        contentType: "application/json",
        dataType: 'json'
    });

  }
  else {
    printIntentResponse( "<strong>Type or paste some user text to parse</strong>." );
  }
});