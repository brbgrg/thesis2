$aux_dir = "build";
$out_dir = "build";

$target_pdf_dir = 'C:/Users/barbo/Desktop/thesis repo clone/thesis/Thesis Draft';


$bibtex_use = 2;
$biber = "biber --input-directory=build --output-directory=build %O %S";

$pdf_mode = 1;
$postscript = "move build\\*.pdf \"$target_pdf_dir\"";

