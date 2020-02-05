$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    
    
    var sym = ['No Symptoms exist for Healthy Plants','The first symptoms of tropical rust appear on older leaves. They are mostly visible in the form of minute, bright yellow to orange lesions on the upper leaf surface. On the underside, spots of similar color but slightly bigger and rougher in appearance appear. As the disease progresses, they turn into target, raised, pale brown fungal pustules surrounded by a yellow halo. As they burst open and release their spores, they often coalesce and fom irregular dark brown spots. On stems and petioles these pustules are usually elongated and are not much raised. Plants become prematurely defoliated as the disease develops, resulting in reduced boll size. ','White hairy insects can be feeding on buds, twigs, branches, and even on the roots. Distorted leaves, yellowing foliage, poor growth and branch dieback are the consequence of this activity. A white fluffy covering and honeydew appear near the feeding sites. On bark and shoots, the development of cankers and swellings is also characteristic. Underground forms of the aphid also attack roots and lead to the formation of swollen enlargements or large knots. The impaired transport of water and nutrients explains the yellowish appearance of the the trees. These galls increase in size from year to year as a result of aphid feeding The wounds caused by the insects and the presence of the honeydew also attract opportunistic fungi that may cover the infected tissues with sooty mold. Young trees are easily uprooted when infested','Initially, trifoliate leaves become slightly lighter in color. Gradually, light and dark green mosaic pattern appears on the leaf blade ( o green mosaic). Some veins or parts of them show sign of chlorosis (yellowing). As the disease progresses, parts of the leaves might become puckered, blistered or distorted Curled down or rolled leaves are other late symptoms. Plants which were infected during the early growth stages may be severely stunted and unproductive, with fewer pods and fewer seeds per pod in some susceptible varieties, the virus can cause the blackening of roots, a symptom that is only observed at temperatures above 30 C '];
    
    var tri = ['Healthy Crops have no triggering factors','Rust of cotton is an aggressive disease caused by the fungus Phakopsora gossypii. It is not seed- or soil-borne and thus needs green living tissue to survive. During the season, the spores produced in the pustules of cotton infect gramma grasses (Bouteloua spp) around the fields and produce elongate brownish or black spots on their leaves. At the beginning of the next season it is the spores produced on these grasses that will infect cotton plants to complete the cycle. The spores penetrate the plant cells directly rather than through pores or wounds in the leaf tissue. High humidity leaf wetness and moderate to warm temperatures are conducive for the disease','The symptoms are caused by the feeding activity of the woolly aphid Eriosoma lanigerum. Unlike most aphids, it sucks sap from the woody stems, rather than from foliage. This insect is characterized by its white, thick, fluffy wax covering. It overwinters on its host in cracks on the bark or on suppressed wounds around old feeding sites. As temperature increases in the spring the aphids become active again and climb on the suckers, younger shoots and branches in search of a vulnerable sites (areas with thinner bark). There, it feeds gregariously, sucking sap from beneath the bark, and starts secreting the fluffy hairs that eventually wrapped the colony. Opportunistic pathogen con the colonize the open wounds. During the cure, the adults grow wings and any off in search of new host plants Elm trees in the vicinity of orchards increase the migration of the aphid to the apple orchards.','The primary source of inoculum are infected seeds. The secondary transmission from plant to plant occurs via infected pollen, vectors pests (mostly aphids) or through mechanical injury to the plants during field work Symptoms and effects on yield depends on plant variety, environmental conditions (temperature and humidity) and time of infection. Runner beans seem to be immune to the virus, while pole beans and bush beans are more vulnerable Losses of up to 100% can be found in susceptible plants grown from seeds carrying the virus (seed-borne infection). Later infections by aphids are usually less severe.'];
    
    var pre = ['Adjust your planting depth.Manage varieties according to vigor.Manage seeding rates for planting date, environmental conditions, soil types, etc. Ensure that liquid in-furrow insecticides directly contact seed and good coverage is achieved.','Plant early and if possible choose an early maturing cultivar.  Alternatively plant late to take advantage of drier periods  Use wider row spacing to hasten canopy drying.  Monitor your plants regularly and weed out alternative hosts. particularly gramma grasses','Choose resistant varieties if available. In mild infestations, the insect can be monitored and scrubbedwith a brush. Apply fortifiers or balanced fertilization to strengthen the trees. Avoid the excessive usage of pesticides as they can decreasepopulation of beneficial insects. Summer prune in late summer to remove developing colonies. Remove affected young shoots and branches. Remove suckers at the base of the tree to create less favorableconditions for the aphids. Paint large pruning cuts with commercial pruning paint todiscourage aphid colonies. Do not plant elm trees close to apple tree orchard.','Use healthy seeding material from certified sources. Plant resilient varieties whenever possible Plant densely to prevent aphids from entering the canopy. Plant early to avoid aphid peak population. Remove infected plants when first symptoms are observed Cultivate beans far from other bean production sites. Rotate crops with non-host plants Plant companion crops to block the aphids'];

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(data);
                $('#s').text('Symptoms:');
                $('#t').text('Triggers:');
                $('#pr').text('Precautions:');
                if(data == "Rust"){
                  $('#sym').text(sym[1]);
                  $('#tri').text(tri[1]);
                  $('#pre').text(pre[1]);
                }
              else if(data == "Wooly Aphids"){
                $('#sym').text(sym[2]);
                $('#tri').text(tri[2]);
                $('#pre').text(pre[2]);
              }
              else if(data == "Mosaic Virus"){
                $('#sym').text(sym[3]);
                $('#tri').text(tri[3]);
                $('#pre').text(pre[3]);
              }
              else{
                $('#sym').text(sym[0]);
                $('#tri').text(tri[0]);
                $('#pre').text(pre[0]);
              }
                console.log('Success!');
            },
        });
    });

});
