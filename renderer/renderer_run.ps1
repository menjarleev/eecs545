$blender_path = 'C:\Program Files\Blender Foundation\Blender 2.92\blender.exe'
$model_id = '02747177'
$model_id_path = [string]::Format('..\ShapeNetCore.v2\{0}\',$model_id)

$items = Get-ChildItem $model_id_path

foreach ($item in $items){
    & $blender_path --background --python ./render_blender.py -- --output_folder ./output/$model_id/$item/ $model_id_path/$item/models/model_normalized.obj --resolution 256
}
