// Ayam Serayu – Main JS
console.log('🍗 Sistem Prediksi Penjualan Ayam Serayu Bekasi loaded');

// Auto-dismiss alerts after 5s
document.querySelectorAll('.alert').forEach(el => {
  setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity 0.5s'; }, 5000);
});
